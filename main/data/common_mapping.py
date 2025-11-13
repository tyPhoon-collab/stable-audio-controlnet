"""
共通のマッピングクラス定義
"""

import csv


class BaseMapping:
    """CSVマッピングのベースクラス（シングルトン）"""

    _instance = None
    _mapping = None
    _csv_path = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_csv_column_names(self) -> tuple[str, str]:
        """サブクラスでオーバーライド: (key_column, value_column)を返す"""
        raise NotImplementedError

    def load_mapping(self, csv_path: str) -> dict[str, str]:
        """CSVファイルからマッピングを読み込む"""
        if (
            getattr(self, "_mapping", None) is not None
            and getattr(self, "_csv_path", None) == csv_path
        ):
            return getattr(self, "_mapping", {})

        key_col, value_col = self._get_csv_column_names()
        mapping: dict[str, str] = {}

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                track_key = row.get(key_col, "").strip()
                value = (row.get(value_col) or "").strip()
                if not track_key or not value:
                    continue
                mapping[track_key] = value

        self._mapping = mapping
        self._csv_path = csv_path
        return mapping

    def get_value(self, track_name: str, default: str = "") -> str:
        """トラック名からマッピング値を取得"""
        mapping = getattr(self, "_mapping", None)
        if mapping is None:
            return default
        return mapping.get(track_name, default)


class GenreMapping(BaseMapping):
    """ジャンル情報マッピング"""

    _instance = None

    def _get_csv_column_names(self) -> tuple[str, str]:
        return ("Track Name", "Genre")

    def get_genre(self, track_name: str, default: str = "Unknown") -> str:
        """トラック名からジャンルを取得"""
        return self.get_value(track_name, default)


class DescriptionMapping(BaseMapping):
    """説明文マッピング"""

    _instance = None

    def _get_csv_column_names(self) -> tuple[str, str]:
        return ("key", "description")

    def get_description(self, track_name: str, default: str = "") -> str:
        """トラック名から説明文を取得"""
        return self.get_value(track_name, default)
