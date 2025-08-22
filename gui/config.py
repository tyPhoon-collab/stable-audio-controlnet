"""
GUI アプリケーション用の設定

このファイルでGUIアプリケーションの各種設定を管理します。
"""

# デフォルト設定
DEFAULT_CONFIG = {
    # モデル設定
    "model": {
        "checkpoint_path": "../ckpts/musdb-chord/last.ckpt",
        "config_name": "train_musdb_controlnet_chord",
        "sample_rate": 44100,
    },
    # 生成設定
    "generation": {
        "default_prompt": "A beautiful piano composition",
        "default_bpm": 120,
        "default_key_root": "C",
        "default_key_mode": "major",
        "default_sampling_steps": 50,
        "default_cfg_scale": 7.0,
        "default_seed": 42,
        "sigma_min": 0.3,
        "sigma_max": 500,
        "sampler_type": "dpmpp-3m-sde",
    },
    # UI設定
    "ui": {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "share": False,
        "show_error": True,
    },
    # コード進行設定
    "chords": {
        "default_duration": 2.0,
        "frame_rate": 100.0,
        "chord_roots": [
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
        "chord_qualities": [
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
        "inversions": ["0", "1", "2", "3"],
        "keys": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        "modes": ["major", "minor"],
    },
    # プリセット進行
    "presets": {
        "royal_road": {
            "name": "王道進行 (vi-IV-I-V)",
            "description": "ポップスでよく使われる定番進行",
            "chords": [
                ("A", "min", 0, 2.0),
                ("F", "maj", 0, 2.0),
                ("C", "maj", 0, 2.0),
                ("G", "maj", 0, 2.0),
            ],
        },
        "canon": {
            "name": "カノン進行 (I-V-vi-iii-IV-I-IV-V)",
            "description": "パッヘルベルのカノンで有名な美しい進行",
            "chords": [
                ("C", "maj", 0, 1.0),
                ("G", "maj", 0, 1.0),
                ("A", "min", 0, 1.0),
                ("E", "min", 0, 1.0),
                ("F", "maj", 0, 1.0),
                ("C", "maj", 0, 1.0),
                ("F", "maj", 0, 1.0),
                ("G", "maj", 0, 1.0),
            ],
        },
        "circle": {
            "name": "循環コード (I-vi-ii-V)",
            "description": "ジャズでよく使われる循環進行",
            "chords": [
                ("C", "maj", 0, 2.0),
                ("A", "min", 0, 2.0),
                ("D", "min", 0, 2.0),
                ("G", "maj", 0, 2.0),
            ],
        },
        "blues": {
            "name": "ブルース進行",
            "description": "12小節のブルース進行",
            "chords": [
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
            ],
        },
    },
}


def get_config():
    """設定を取得"""
    return DEFAULT_CONFIG


def get_model_config():
    """モデル設定を取得"""
    return DEFAULT_CONFIG["model"]


def get_generation_config():
    """生成設定を取得"""
    return DEFAULT_CONFIG["generation"]


def get_ui_config():
    """UI設定を取得"""
    return DEFAULT_CONFIG["ui"]


def get_chord_config():
    """コード設定を取得"""
    return DEFAULT_CONFIG["chords"]


def get_preset_config():
    """プリセット設定を取得"""
    return DEFAULT_CONFIG["presets"]
