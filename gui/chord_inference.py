import os
import sys
import tempfile
from typing import Any, Dict, List

import hydra
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from stable_audio_tools.inference.generation import generate_diffusion_cond

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from main.data.annotation import ChordAnnotation


class ChordInferenceEngine:
    """Stable Audio ControlNet Chord モデルの推論エンジン"""

    def __init__(self):
        self.model = None
        self.model_config = None
        self.sample_rate = 44100
        self.sample_size = None
        self.chord_frame_rate = 24.0  # デフォルト値
        self.chord_annotation = ChordAnnotation(sample_rate=self.sample_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    def load_model(
        self,
        checkpoint_path: str,
        config_path: str = "../exp/train_musdb_controlnet_chord.yaml",
    ):
        """モデルとコンフィグを読み込む"""
        try:
            # Hydraでコンフィグ読み込み
            with hydra.initialize(config_path="..", version_base=None):
                cfg = hydra.compose(
                    config_name="config", overrides=["exp=train_musdb_controlnet_chord"]
                )

            # モデル初期化
            self.model = hydra.utils.instantiate(cfg["model"])

            # チェックポイント読み込み
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                self.model.load_state_dict(ckpt["state_dict"], strict=False)
                print(f"✅ チェックポイントを読み込みました: {checkpoint_path}")
            else:
                print(
                    f"⚠️ チェックポイントが見つかりません。プリトレインモデルを使用: {checkpoint_path}"
                )

            self.model = self.model.to(self.device)
            self.model.eval()

            # モデル設定を取得
            self.sample_rate = self.model.sample_rate
            self.sample_size = self.model.sample_size
            self.model_config = self.model.model_config

            # 設定からchord_frame_rateを取得
            if hasattr(cfg, "chord_frame_rate"):
                self.chord_frame_rate = cfg.chord_frame_rate
            else:
                self.chord_frame_rate = 24.0  # デフォルト値

            self._loaded = True
            print("✅ モデル読み込み完了")
            print(f"   - Sample Rate: {self.sample_rate}")
            print(f"   - Sample Size: {self.sample_size}")
            print(f"   - Chord Frame Rate: {self.chord_frame_rate}")
            print(f"   - Device: {self.device}")

        except Exception as e:
            print(f"❌ モデル読み込みエラー: {str(e)}")
            raise e

    def is_loaded(self) -> bool:
        """モデルが読み込まれているかチェック"""
        return self._loaded and self.model is not None

    def _chord_sequence_to_tensor(
        self,
        chord_sequence: List[List],
        total_duration: float,
    ) -> torch.Tensor:
        """
        コードシーケンスをテンソルに変換

        Args:
            chord_sequence: [[root, quality, inversion, duration], ...] のリスト
            total_duration: 総時間（秒）

        Returns:
            shape: (num_frames, 3) の tensor [root, quality, inversion]
        """
        num_frames = int(total_duration * self.chord_frame_rate)
        chord_tensor = torch.full((num_frames, 3), -1, dtype=torch.long)

        current_time = 0.0
        for root_str, quality_str, inversion, duration in chord_sequence:
            # 文字列から数値に変換
            root = self.chord_annotation.CHORD_ROOT_MAP.get(root_str, -1)
            quality = self.chord_annotation.CHORD_QUALITY_MAP.get(quality_str, -1)

            start_frame = int(current_time * self.chord_frame_rate)
            end_frame = int((current_time + duration) * self.chord_frame_rate)

            # フレーム範囲のクリッピング
            start_frame = max(0, min(start_frame, num_frames))
            end_frame = max(0, min(end_frame, num_frames))

            if start_frame < end_frame:
                chord_tensor[start_frame:end_frame] = torch.tensor(
                    [root, quality, inversion]
                )

            current_time += duration

        return chord_tensor

    def _chord_to_onehot(self, chord_batch: torch.Tensor) -> torch.Tensor:
        """
        コードテンソルをone-hotエンコーディングに変換
        （module_controlnet_chord.py の実装をコピー）
        """
        B, T, _ = chord_batch.shape
        device = chord_batch.device

        root = chord_batch[..., 0].clone()
        qual = chord_batch[..., 1].clone()
        inv = chord_batch[..., 2].clone()

        # Map -1 to last index in its group
        root_idx = torch.where(
            root >= 0, root, torch.full_like(root, 12)
        )  # 0..11, 12 for N
        qual_idx = torch.where(
            qual >= 0, qual, torch.full_like(qual, 9)
        )  # 0..8, 9 for N
        # inversion: 0..6 valid, others -> 7
        inv_idx = torch.where((inv >= 0) & (inv <= 6), inv, torch.full_like(inv, 7))

        C_root, C_qual, C_inv = 13, 10, 8
        C_total = C_root + C_qual + C_inv
        out = torch.zeros((B, C_total, T), device=device, dtype=torch.float32)

        # scatter for each group
        # root
        out_root = out[:, 0:C_root]
        out_root.zero_()
        out_root.scatter_(1, root_idx.long().unsqueeze(1), 1.0)

        # quality
        out_qual = out[:, C_root : C_root + C_qual]
        out_qual.zero_()
        out_qual.scatter_(1, qual_idx.long().unsqueeze(1), 1.0)

        # inversion
        out_inv = out[:, C_root + C_qual :]
        out_inv.zero_()
        out_inv.scatter_(1, inv_idx.long().unsqueeze(1), 1.0)

        return out

    def generate_from_chord_sequence(
        self, chord_sequence: List[List], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        コードシーケンスから音楽を生成

        Args:
            chord_sequence: [[root, quality, inversion, duration], ...] のリスト
            params: 生成パラメータ

        Returns:
            生成結果の辞書
        """
        if not self.is_loaded():
            raise RuntimeError("モデルが読み込まれていません")

        # シード設定
        torch.manual_seed(params["seed"])
        np.random.seed(params["seed"])

        # 総時間計算
        total_duration = sum(chord[3] for chord in chord_sequence)

        # コードシーケンスをテンソルに変換（設定されたchord_frame_rateを使用）
        chord_tensor = self._chord_sequence_to_tensor(
            chord_sequence, total_duration
        ).unsqueeze(0)  # バッチ次元追加

        # one-hotエンコーディング
        chord_onehot = self._chord_to_onehot(chord_tensor.to(self.device))

        # サンプル数計算
        num_samples = int(total_duration * self.sample_rate)

        # コードを音声サンプル長にリサイズ
        chord_rescaled = F.interpolate(chord_onehot, size=num_samples, mode="nearest")

        # 条件付けデータ作成
        conditioning = [
            {
                "prompt": params["prompt"],
                "seconds_start": 0.0,
                "seconds_total": total_duration,
                "chord": chord_rescaled[0:1],  # バッチサイズ1
            }
        ]

        print("🎵 音楽生成中...")
        print(f"   - 総時間: {total_duration:.2f}秒")
        print(f"   - コード数: {len(chord_sequence)}")
        print(f"   - プロンプト: {params['prompt']}")
        print(f"   - 音声サンプルレート: {self.sample_rate}Hz")
        print(f"   - コードフレームレート: {self.chord_frame_rate}Hz")

        # 音楽生成
        with torch.no_grad():
            output = generate_diffusion_cond(
                self.model.model,
                batch_size=1,
                steps=params["sampling_steps"],
                cfg_scale=params["cfg_scale"],
                conditioning=conditioning,
                sample_size=self.sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=self.device,
            )

        # 音声を正規化
        audio = torch.clamp(output[0], -1.0, 1.0).cpu().numpy()

        # 一時ファイルに保存
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=".")
        temp_path = temp_file.name
        temp_file.close()

        # WAVファイルとして保存
        import soundfile as sf

        sf.write(temp_path, audio, self.sample_rate)

        # スペクトログラム生成
        spectrogram_fig = self._create_spectrogram(audio, self.sample_rate)

        # 生成情報
        info = {
            "total_duration": total_duration,
            "num_chords": len(chord_sequence),
            "chord_sequence": [
                f"{chord[0]}:{chord[1]}" + (f"/{chord[2]}" if chord[2] > 0 else "")
                for chord in chord_sequence
            ],
            "parameters": params,
            "sample_rate": self.sample_rate,
            "chord_frame_rate": self.chord_frame_rate,
            "audio_length": len(audio),
            "device": str(self.device),
        }

        print("✅ 生成完了!")

        return {
            "audio_path": temp_path,
            "audio_array": audio,
            "spectrogram": spectrogram_fig,
            "info": info,
        }

    def _create_spectrogram(self, audio: np.ndarray, sr: int):
        """スペクトログラムを作成"""
        plt.figure(figsize=(12, 6))

        # メルスペクトログラム計算
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, fmax=sr // 2
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # プロット
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("メルスペクトログラム")
        plt.xlabel("時間 (秒)")
        plt.ylabel("周波数 (メル)")
        plt.tight_layout()

        return plt.gcf()
