from typing import Any, Dict, List


class ChordSequencer:
    """コード進行のシーケンス管理クラス"""

    def __init__(self):
        self.chord_sequence = []
        self.total_duration = 0.0

    def add_chord(
        self, root: str, quality: str, inversion: int = 0, duration: float = 2.0
    ):
        """コードを追加"""
        chord = [root, quality, inversion, duration]
        self.chord_sequence.append(chord)
        self.total_duration += duration
        return chord

    def remove_chord(self, index: int):
        """指定インデックスのコードを削除"""
        if 0 <= index < len(self.chord_sequence):
            removed_chord = self.chord_sequence.pop(index)
            self.total_duration -= removed_chord[3]
            return removed_chord
        return None

    def clear(self):
        """すべてのコードをクリア"""
        self.chord_sequence = []
        self.total_duration = 0.0

    def get_sequence(self) -> List[List]:
        """現在のコードシーケンスを取得"""
        return self.chord_sequence.copy()

    def get_total_duration(self) -> float:
        """総時間を取得"""
        return self.total_duration

    def set_chord_duration(self, index: int, new_duration: float):
        """指定インデックスのコードの時間を変更"""
        if 0 <= index < len(self.chord_sequence):
            old_duration = self.chord_sequence[index][3]
            self.chord_sequence[index][3] = new_duration
            self.total_duration += new_duration - old_duration

    def get_chord_at_time(self, time: float) -> Dict[str, Any]:
        """指定時刻のコード情報を取得"""
        current_time = 0.0
        for i, (root, quality, inversion, duration) in enumerate(self.chord_sequence):
            if current_time <= time < current_time + duration:
                return {
                    "index": i,
                    "root": root,
                    "quality": quality,
                    "inversion": inversion,
                    "duration": duration,
                    "start_time": current_time,
                    "end_time": current_time + duration,
                }
            current_time += duration
        return None

    def to_timeline_data(self) -> List[Dict[str, Any]]:
        """タイムライン表示用のデータを生成"""
        timeline = []
        current_time = 0.0

        for i, (root, quality, inversion, duration) in enumerate(self.chord_sequence):
            chord_name = f"{root}:{quality}"
            if inversion > 0:
                chord_name += f"/{inversion}"

            timeline.append(
                {
                    "index": i,
                    "chord_name": chord_name,
                    "root": root,
                    "quality": quality,
                    "inversion": inversion,
                    "duration": duration,
                    "start_time": current_time,
                    "end_time": current_time + duration,
                }
            )
            current_time += duration

        return timeline

    @staticmethod
    def create_preset_progression(preset_name: str) -> List[List]:
        """プリセット進行を作成"""
        presets = {
            "royal_road": [
                ("A", "min", 0, 2.0),
                ("F", "maj", 0, 2.0),
                ("C", "maj", 0, 2.0),
                ("G", "maj", 0, 2.0),
            ],
            "canon": [
                ("C", "maj", 0, 1.0),
                ("G", "maj", 0, 1.0),
                ("A", "min", 0, 1.0),
                ("E", "min", 0, 1.0),
                ("F", "maj", 0, 1.0),
                ("C", "maj", 0, 1.0),
                ("F", "maj", 0, 1.0),
                ("G", "maj", 0, 1.0),
            ],
            "circle": [
                ("C", "maj", 0, 2.0),
                ("A", "min", 0, 2.0),
                ("D", "min", 0, 2.0),
                ("G", "maj", 0, 2.0),
            ],
            "blues": [
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
        }

        return list(presets.get(preset_name, []))

    @staticmethod
    def chord_symbol_to_parts(chord_symbol: str):
        """コード記号を分解 (例: "C:maj/2" -> ("C", "maj", 2))"""
        inversion = 0

        if "/" in chord_symbol:
            chord_symbol, inversion_str = chord_symbol.split("/")
            try:
                inversion = int(inversion_str)
            except ValueError:
                inversion = 0

        if ":" in chord_symbol:
            root, quality = chord_symbol.split(":")
        else:
            root = chord_symbol
            quality = "maj"

        return root, quality, inversion

    @staticmethod
    def parts_to_chord_symbol(root: str, quality: str, inversion: int = 0) -> str:
        """コード要素からコード記号を作成"""
        chord_symbol = f"{root}:{quality}"
        if inversion > 0:
            chord_symbol += f"/{inversion}"
        return chord_symbol
