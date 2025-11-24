import typing as tp

import torch
from einops import rearrange
from stable_audio_tools.models.blocks import FourierFeatures
from stable_audio_tools.models.transformer import ContinuousTransformer
from torch import nn
from torch.nn import functional as F
from x_transformers import ContinuousTransformerWrapper, Encoder

# from main.controlnet.conditioning import ControlNetConditioningEmbedding


class ControlNetDiffusionTransformer(nn.Module):
    def __init__(
        self,
        io_channels=32,
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal[
            "x-transformers", "continuous_transformer"
        ] = "x-transformers",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        **kwargs,
    ):
        super().__init__()

        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        timestep_features_dim = 256

        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        if cond_token_dim > 0:
            # Conditioning tokens

            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False),
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False),
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False),
            )

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size

        # Transformer

        self.transformer_type = transformer_type

        self.global_cond_type = global_cond_type

        if self.transformer_type == "x-transformers":
            self.transformer = ContinuousTransformerWrapper(
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                max_seq_len=0,  # Not relevant without absolute positional embeds
                attn_layers=Encoder(
                    dim=embed_dim,
                    depth=depth,
                    heads=num_heads,
                    attn_flash=True,
                    cross_attend=cond_token_dim > 0,
                    dim_context=None if cond_embed_dim == 0 else cond_embed_dim,
                    zero_init_branch_output=True,
                    use_abs_pos_emb=False,
                    rotary_pos_emb=True,
                    ff_swish=True,
                    ff_glu=True,
                    **kwargs,
                ),
            )

        elif self.transformer_type == "continuous_transformer":
            global_dim = None

            if self.global_cond_type == "adaLN":
                # The global conditioning is projected to the embed_dim already at this point
                global_dim = embed_dim

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                cross_attend=cond_token_dim > 0,
                cond_token_dim=cond_embed_dim,
                global_cond_dim=global_dim,
                **kwargs,
            )

        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)

        # controlnet stuff

        self.conv_in = nn.Conv1d(dim_in, dim_in, 1)
        nn.init.zeros_(self.conv_in.weight)
        nn.init.zeros_(self.conv_in.bias)

        self.conv_outs = nn.ModuleList(
            [nn.Conv1d(embed_dim, embed_dim, 1) for _ in range(depth)]
        )
        for conv_out in self.conv_outs:
            nn.init.zeros_(conv_out.weight)
            nn.init.zeros_(conv_out.bias)

    def _forward(
        self,
        x,
        t,
        controlnet_cond=None,
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=None,
        **kwargs,
    ):
        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)

            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

        if input_concat_cond is not None:
            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(
                    input_concat_cond, (x.shape[2],), mode="nearest"
                )

            x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(
            self.timestep_features(t[:, None])
        )  # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists
        if global_embed is not None:
            global_embed = global_embed + timestep_embed
        else:
            global_embed = timestep_embed

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend":
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones(
                    (x.shape[0], 1), device=x.device, dtype=torch.bool
                )
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat(
                    [prepend_inputs, global_embed.unsqueeze(1)], dim=1
                )
                prepend_mask = torch.cat(
                    [
                        prepend_mask,
                        torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool),
                    ],
                    dim=1,
                )

            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x
        controlnet_cond = self.conv_in(controlnet_cond)
        x = x + controlnet_cond

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        if self.transformer_type == "continuous_transformer":
            _, info = self.transformer(
                x,
                prepend_embeds=prepend_inputs,
                context=cross_attn_cond,
                context_mask=cross_attn_cond_mask,
                mask=mask,
                prepend_mask=prepend_mask,
                return_info=True,
                **extra_args,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                f"Unknown transformer type: {self.transformer_type}"
            )

        out_info = []
        for i, conv_out in enumerate(self.conv_outs):
            h = rearrange(info["hidden_states"][i], "b t c -> b c t")
            h = rearrange(conv_out(h), "b c t -> b t c")
            out_info.append(h)
        return out_info

    def forward(
        self,
        x,
        t,
        controlnet_cond=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        causal=False,
        cfg_dropout_prob=0.0,
        cfg_scale=1.0,
        mask=None,
        **kwargs,
    ):
        assert causal == False, "Causal mode is not supported for DiffusionTransformer"

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()

            cross_attn_cond_mask = None  # Temporarily disabling conditioning masks due to kernel issue for flash attention

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        cross_attn_dropout_mask = None
        prepend_dropout_mask = None
        if cfg_dropout_prob > 0.0:
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(
                    cross_attn_cond, device=cross_attn_cond.device
                )
                cross_attn_dropout_mask = torch.bernoulli(
                    torch.full(
                        (cross_attn_cond.shape[0], 1, 1),
                        cfg_dropout_prob,
                        device=cross_attn_cond.device,
                    )
                ).to(torch.bool)
                cross_attn_cond = torch.where(
                    cross_attn_dropout_mask, null_embed, cross_attn_cond
                )

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                prepend_dropout_mask = torch.bernoulli(
                    torch.full(
                        (prepend_cond.shape[0], 1, 1),
                        cfg_dropout_prob,
                        device=prepend_cond.device,
                    )
                ).to(torch.bool)
                prepend_cond = torch.where(
                    prepend_dropout_mask, null_embed, prepend_cond
                )

        # batch_cfg=Trueを前提とする（diffusion.pyで強制されている）
        # controlnet_condをxのバッチサイズに合わせる
        if controlnet_cond is not None and controlnet_cond.shape[0] != x.shape[0]:
            repeat_factor = x.shape[0] // controlnet_cond.shape[0]
            if (
                repeat_factor * controlnet_cond.shape[0] == x.shape[0]
                and repeat_factor > 0
            ):
                controlnet_cond = controlnet_cond.repeat(repeat_factor, 1, 1)

        # batch_cfg=Trueの場合、CFG処理は外部で既に行われているため、
        # ここでは単純に_forwardを呼び出すだけ
        return (
            self._forward(
                x,
                t,
                controlnet_cond=controlnet_cond,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                **kwargs,
            ),
            cross_attn_dropout_mask,
            prepend_dropout_mask,
        )
