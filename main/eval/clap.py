import functools

import laion_clap
import librosa
import numpy as np
import torch


def initialize_clap_model(
    weights_path: str = "ckpts/music_audioset_epoch_15_esc_90.14.pt",
    amodel: str = "HTSAT-base",
    device: str = "cuda",
) -> laion_clap.CLAP_Module:
    """
    CLAPモデルを初期化し、重みをロードして返します。
    """
    # モデルのロード
    model = laion_clap.CLAP_Module(
        enable_fusion=False,
        device=device,
        amodel=amodel,
    )

    # オリジナルの関数を保存
    _original_torch_load = torch.load

    # ラッパー関数を作成（常に weights_only=False を強制）
    @functools.wraps(_original_torch_load)
    def unsafe_torch_load(*args, **kwargs):
        # weights_onlyが指定されていない、またはTrueの場合でもFalseに上書き
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    # torch.load を置き換え
    torch.load = unsafe_torch_load
    model.load_ckpt(weights_path)
    torch.load = _original_torch_load

    model.to(device)
    model.eval()

    return model


def calculate_pair_similarity(
    model: laion_clap.CLAP_Module,
    audio_input: str | np.ndarray,
    text: str,
    device: str = "cuda",
) -> float:
    """
    単一のオーディオと単一のテキストのコサイン類似度を計算します。
    モデルの読み込みは重たいので、事前に initialize_clap_model で行い、再利用してください。

    Returns:
        float: コサイン類似度 (-1.0 ~ 1.0)
    """
    # 1. オーディオのロード・前処理 (48kHz必須)
    audio_data: np.ndarray
    if isinstance(audio_input, str):
        # audio_input is a file path
        audio_data, _ = librosa.load(audio_input, sr=48000)
    else:
        # audio_input is raw data
        audio_data = audio_input

    # (Batch, Time) 形状に調整
    audio_data = audio_data.reshape(1, -1)

    # 2. 推論 (Embedding)
    # get_text_embedding はリスト入力を期待するため [text] でラップします
    with torch.no_grad():
        text_embed_np = model.get_text_embedding([text])
        audio_embed_np = model.get_audio_embedding_from_data(
            x=audio_data, use_tensor=False
        )

        # Numpy -> Tensor & Device転送
        text_tensor = torch.from_numpy(text_embed_np).to(device)
        audio_tensor = torch.from_numpy(audio_embed_np).to(device)

        # 3. 正規化 (L2 Norm)
        text_tensor = text_tensor / text_tensor.norm(dim=-1, keepdim=True)
        audio_tensor = audio_tensor / audio_tensor.norm(dim=-1, keepdim=True)

        # 4. スコア計算 (Cosine Similarity)
        # 1対1なので行列積の結果は (1, 1) になる -> item() で float 化
        similarity = (text_tensor @ audio_tensor.T).item()

    return similarity


if __name__ == "__main__":
    # 1. 初期化
    model = initialize_clap_model()

    # 2. テストデータ
    description = "An upbeat track featuring acoustic guitars, rhythmic percussion, and bright piano, creating a lively and cheerful mood with a medium tempo."
    audio_path = "data/Al James - Schoolboy Facination_no_vocals.mp3"

    # 3. 個別に実行
    score = calculate_pair_similarity(model, audio_path, description)

    # 4. 結果表示
    print("\n--- Single Pair Comparison Results ---")
    print(f"Score: {score:.4f}")
