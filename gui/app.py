import os
import sys
from typing import List

import gradio as gr
import numpy as np

# プロジェクトのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from chord_inference import ChordInferenceEngine
    from chord_sequencer import ChordSequencer
    from config import get_config, get_preset_config

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 依存関係のインポートエラー: {e}")
    print("pip install -r requirements_gui.txt を実行してください")
    DEPENDENCIES_AVAILABLE = False

    # フォールバック用のダミークラス
    class ChordInferenceEngine:
        def __init__(self):
            pass

        def is_loaded(self):
            return False

        def load_model(self, path):
            return "❌ 依存関係が不足しています"

        def generate_from_chord_sequence(self, seq, params):
            raise RuntimeError("依存関係が不足しています")

    class ChordSequencer:
        def __init__(self):
            pass


class ChordProgressionGUI:
    def __init__(self):
        self.inference_engine = ChordInferenceEngine()
        self.chord_sequencer = ChordSequencer()
        self.current_sequence = []

    def create_interface(self):
        """Gradio インターフェースを作成"""

        with gr.Blocks(
            title="Stable Audio ControlNet - Chord Progression Generator",
            theme=gr.themes.Soft(),
        ) as interface:
            gr.Markdown("# 🎵 Stable Audio ControlNet - Chord Progression Generator")
            gr.Markdown("シーケンシャルな和音進行入力で音楽を生成します")

            with gr.Row():
                # 左側: モデル設定とコード進行エディター
                with gr.Column(scale=1):
                    gr.Markdown("## ⚙️ モデル設定")

                    model_path = gr.Textbox(
                        label="モデルチェックポイントパス",
                        value="../ckpts/musdb-chord/last.ckpt",
                        placeholder="モデルファイルのパスを入力",
                    )

                    load_model_btn = gr.Button("🔄 モデル読み込み", variant="primary")
                    model_status = gr.Textbox(
                        label="モデル状態", value="未読み込み", interactive=False
                    )

                    gr.Markdown("## 🎹 和音進行エディター")

                    # BPMとキー設定
                    with gr.Row():
                        bpm = gr.Slider(60, 200, value=120, step=1, label="BPM")
                        key_root = gr.Dropdown(
                            choices=[
                                "C",
                                "C#",
                                "D",
                                "D#",
                                "E",
                                "F",
                                "F#",
                                "G",
                                "G#",
                                "A",
                                "A#",
                                "B",
                            ],
                            value="C",
                            label="キー",
                        )
                        key_mode = gr.Dropdown(
                            choices=["major", "minor"], value="major", label="モード"
                        )

                    # 和音入力セクション
                    with gr.Row():
                        chord_root = gr.Dropdown(
                            choices=[
                                "C",
                                "C#",
                                "D",
                                "D#",
                                "E",
                                "F",
                                "F#",
                                "G",
                                "G#",
                                "A",
                                "A#",
                                "B",
                            ],
                            value="C",
                            label="コードルート",
                        )
                        chord_quality = gr.Dropdown(
                            choices=[
                                "maj",
                                "min",
                                "maj7",
                                "min7",
                                "dom7",
                                "dim",
                                "aug",
                                "sus4",
                                "sus2",
                            ],
                            value="maj",
                            label="コード種類",
                        )
                        chord_inversion = gr.Dropdown(
                            choices=["0", "1", "2", "3"], value="0", label="転回形"
                        )
                        chord_duration = gr.Slider(
                            0.5, 8.0, value=2.0, step=0.5, label="長さ(秒)"
                        )

                    with gr.Row():
                        add_chord_btn = gr.Button("➕ コード追加", variant="secondary")
                        clear_sequence_btn = gr.Button(
                            "🗑️ リセット", variant="secondary"
                        )

                    # 現在のコード進行表示
                    chord_sequence_display = gr.DataFrame(
                        headers=["#", "コード", "長さ(秒)", "開始時間"],
                        datatype=["number", "str", "number", "number"],
                        value=[],
                        label="現在のコード進行",
                        interactive=False,
                        max_height=200,
                    )

                    # プリセットコード進行
                    gr.Markdown("### 🎼 プリセット進行")
                    preset_buttons = [
                        gr.Button("🎵 王道進行 (vi-IV-I-V)", variant="secondary"),
                        gr.Button(
                            "🎵 カノン進行 (I-V-vi-iii-IV-I-IV-V)", variant="secondary"
                        ),
                        gr.Button("🎵 循環コード (I-vi-ii-V)", variant="secondary"),
                        gr.Button("🎵 ブルース進行", variant="secondary"),
                    ]

                # 右側: 生成設定と出力
                with gr.Column(scale=1):
                    gr.Markdown("## 🎧 生成設定")

                    prompt = gr.Textbox(
                        label="プロンプト",
                        value="A beautiful piano composition",
                        placeholder="生成したい音楽のスタイルや楽器を説明",
                    )

                    with gr.Row():
                        sampling_steps = gr.Slider(
                            10, 100, value=50, step=5, label="サンプリングステップ"
                        )
                        cfg_scale = gr.Slider(
                            1.0, 15.0, value=7.0, step=0.5, label="CFGスケール"
                        )

                    with gr.Row():
                        seed = gr.Number(value=42, label="シード値", precision=0)
                        randomize_seed = gr.Checkbox(
                            label="ランダムシード", value=False
                        )

                    generate_btn = gr.Button(
                        "🎵 音楽生成", variant="primary", size="lg"
                    )

                    gr.Markdown("## 📊 生成進捗")
                    progress_bar = gr.Progress()
                    generation_status = gr.Textbox(
                        label="状態", value="待機中", interactive=False
                    )

                    gr.Markdown("## 🔊 生成結果")

                    # 音声出力
                    output_audio = gr.Audio(label="生成された音楽", type="filepath")

                    # スペクトログラム表示
                    spectrogram_plot = gr.Plot(label="スペクトログラム")

                    # 詳細情報
                    with gr.Accordion("📋 生成詳細", open=False):
                        generation_info = gr.JSON(label="生成情報")

            # イベントハンドラーの設定
            load_model_btn.click(
                fn=self.load_model, inputs=[model_path], outputs=[model_status]
            )

            add_chord_btn.click(
                fn=self.add_chord_to_sequence,
                inputs=[chord_root, chord_quality, chord_inversion, chord_duration],
                outputs=[chord_sequence_display],
            )

            clear_sequence_btn.click(
                fn=self.clear_sequence, outputs=[chord_sequence_display]
            )

            # プリセット進行ボタン
            preset_buttons[0].click(
                fn=lambda: self.load_preset_progression("royal_road"),
                outputs=[chord_sequence_display],
            )
            preset_buttons[1].click(
                fn=lambda: self.load_preset_progression("canon"),
                outputs=[chord_sequence_display],
            )
            preset_buttons[2].click(
                fn=lambda: self.load_preset_progression("circle"),
                outputs=[chord_sequence_display],
            )
            preset_buttons[3].click(
                fn=lambda: self.load_preset_progression("blues"),
                outputs=[chord_sequence_display],
            )

            generate_btn.click(
                fn=self.generate_audio,
                inputs=[
                    prompt,
                    bpm,
                    key_root,
                    key_mode,
                    sampling_steps,
                    cfg_scale,
                    seed,
                    randomize_seed,
                ],
                outputs=[
                    output_audio,
                    spectrogram_plot,
                    generation_info,
                    generation_status,
                ],
            )

        return interface

    def load_model(self, model_path: str) -> str:
        """モデルを読み込む"""
        try:
            self.inference_engine.load_model(model_path)
            return "✅ モデル読み込み完了"
        except Exception as e:
            return f"❌ エラー: {str(e)}"

    def add_chord_to_sequence(
        self, root: str, quality: str, inversion: str, duration: float
    ) -> List[List]:
        """コードシーケンスに和音を追加"""
        chord_symbol = f"{root}:{quality}"
        if inversion != "0":
            chord_symbol += f"/{inversion}"

        # 開始時間を計算
        start_time = sum(chord[2] for chord in self.current_sequence)

        self.current_sequence.append([root, quality, int(inversion), duration])

        # 表示用データフォーマット
        display_data = []
        current_time = 0.0
        for i, (r, q, inv, dur) in enumerate(self.current_sequence):
            chord_name = f"{r}:{q}"
            if inv > 0:
                chord_name += f"/{inv}"
            display_data.append([i + 1, chord_name, dur, current_time])
            current_time += dur

        return display_data

    def clear_sequence(self) -> List[List]:
        """コードシーケンスをクリア"""
        self.current_sequence = []
        return []

    def load_preset_progression(self, preset_name: str) -> List[List]:
        """プリセット進行を読み込む"""
        self.current_sequence = []

        if preset_name == "royal_road":
            # vi-IV-I-V (Am-F-C-G in C major)
            chords = [
                ("A", "min", 0, 2.0),
                ("F", "maj", 0, 2.0),
                ("C", "maj", 0, 2.0),
                ("G", "maj", 0, 2.0),
            ]
        elif preset_name == "canon":
            # I-V-vi-iii-IV-I-IV-V (C-G-Am-Em-F-C-F-G in C major)
            chords = [
                ("C", "maj", 0, 1.0),
                ("G", "maj", 0, 1.0),
                ("A", "min", 0, 1.0),
                ("E", "min", 0, 1.0),
                ("F", "maj", 0, 1.0),
                ("C", "maj", 0, 1.0),
                ("F", "maj", 0, 1.0),
                ("G", "maj", 0, 1.0),
            ]
        elif preset_name == "circle":
            # I-vi-ii-V (C-Am-Dm-G in C major)
            chords = [
                ("C", "maj", 0, 2.0),
                ("A", "min", 0, 2.0),
                ("D", "min", 0, 2.0),
                ("G", "maj", 0, 2.0),
            ]
        elif preset_name == "blues":
            # I-I-I-I-IV-IV-I-I-V-IV-I-V (12 bar blues in C)
            chords = [
                ("C", "dom7", 0, 1.0),
                ("C", "dom7", 0, 1.0),
                ("C", "dom7", 0, 1.0),
                ("C", "dom7", 0, 1.0),
                ("F", "dom7", 0, 1.0),
                ("F", "dom7", 0, 1.0),
                ("C", "dom7", 0, 1.0),
                ("C", "dom7", 0, 1.0),
                ("G", "dom7", 0, 1.0),
                ("F", "dom7", 0, 1.0),
                ("C", "dom7", 0, 1.0),
                ("G", "dom7", 0, 1.0),
            ]
        else:
            chords = []

        for root, quality, inversion, duration in chords:
            self.current_sequence.append([root, quality, inversion, duration])

        # 表示用データフォーマット
        display_data = []
        current_time = 0.0
        for i, (r, q, inv, dur) in enumerate(self.current_sequence):
            chord_name = f"{r}:{q}"
            if inv > 0:
                chord_name += f"/{inv}"
            display_data.append([i + 1, chord_name, dur, current_time])
            current_time += dur

        return display_data

    def generate_audio(
        self,
        prompt: str,
        bpm: int,
        key_root: str,
        key_mode: str,
        sampling_steps: int,
        cfg_scale: float,
        seed: int,
        randomize_seed: bool,
    ):
        """音楽を生成"""
        if not self.inference_engine.is_loaded():
            return (
                None,
                None,
                {"error": "モデルが読み込まれていません"},
                "❌ モデル未読み込み",
            )

        if not self.current_sequence:
            return (
                None,
                None,
                {"error": "コード進行が入力されていません"},
                "❌ コード進行なし",
            )

        try:
            if randomize_seed:
                seed = np.random.randint(0, 2**31)

            # 生成パラメータ
            params = {
                "prompt": prompt,
                "bpm": bpm,
                "key_root": key_root,
                "key_mode": key_mode,
                "sampling_steps": sampling_steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
            }

            # 音楽生成
            result = self.inference_engine.generate_from_chord_sequence(
                self.current_sequence, params
            )

            return (
                result["audio_path"],
                result["spectrogram"],
                result["info"],
                "✅ 生成完了",
            )

        except Exception as e:
            return None, None, {"error": str(e)}, f"❌ エラー: {str(e)}"


def main():
    """メイン関数"""
    app = ChordProgressionGUI()
    interface = app.create_interface()

    # 環境変数から設定を取得
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    # 共有リンクを作成してローカルで起動
    interface.launch(
        share=False, server_name=server_name, server_port=server_port, show_error=True
    )


if __name__ == "__main__":
    main()
