"""metrics モジュールのテスト"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from main.data.annotation import LabAnnotation
from main.eval.metrics import (
    DirectoryMetrics,
    FileMetrics,
    annotations_duration,
    evaluate_chord_directory,
    evaluate_chord_pair,
    frame_accuracy,
    frame_root_accuracy,
    lab_to_frame_labels,
)


class TestAnnotationsDuration:
    """annotations_duration のテスト"""

    def test_empty_annotations(self):
        """空のアノテーションの場合"""
        result = annotations_duration([])
        assert result == 0.0

    def test_single_annotation(self):
        """1つのアノテーション"""
        annotations: list[LabAnnotation] = [(0.0, 10.0, "C")]
        result = annotations_duration(annotations)
        assert result == 10.0

    def test_multiple_annotations(self):
        """複数のアノテーション"""
        annotations: list[LabAnnotation] = [
            (0.0, 5.0, "C"),
            (5.0, 15.0, "G"),
            (15.0, 20.0, "Am"),
        ]
        result = annotations_duration(annotations)
        assert result == 20.0


class TestLabToFrameLabels:
    """lab_to_frame_labels のテスト"""

    def test_basic_conversion(self):
        """基本的な変換"""
        annotations: list[LabAnnotation] = [(0.0, 1.0, "C")]
        result = lab_to_frame_labels(annotations, frame_rate=10.0, duration=1.0)
        assert len(result) == 10
        assert all(label == "C" for label in result)

    def test_multiple_segments(self):
        """複数のセグメント"""
        annotations: list[LabAnnotation] = [
            (0.0, 0.5, "C"),
            (0.5, 1.0, "G"),
        ]
        result = lab_to_frame_labels(annotations, frame_rate=10.0, duration=1.0)
        assert len(result) == 10
        assert result[0:5] == ["C"] * 5
        assert result[5:10] == ["G"] * 5

    def test_invalid_frame_rate(self):
        """不正なframe_rate"""
        annotations: list[LabAnnotation] = [(0.0, 1.0, "C")]
        with pytest.raises(ValueError):
            lab_to_frame_labels(annotations, frame_rate=0.0)

    def test_empty_annotations(self):
        """空のアノテーション"""
        result = lab_to_frame_labels([], frame_rate=10.0)
        assert result == []


class TestFrameAccuracy:
    """frame_accuracy のテスト"""

    def test_perfect_match(self):
        """完全一致"""
        predictions = ["C", "C", "G", "G"]
        references = ["C", "C", "G", "G"]
        matches, total, skipped = frame_accuracy(predictions, references)
        assert matches == 4
        assert total == 4
        assert skipped == 0

    def test_partial_match(self):
        """部分一致"""
        predictions = ["C", "C", "G", "Am"]
        references = ["C", "G", "G", "Am"]
        matches, total, skipped = frame_accuracy(predictions, references)
        assert matches == 3  # C(0), G(2), Am(3) が一致
        assert total == 4
        assert skipped == 0

    def test_with_ignore_label(self):
        """ラベルの無視"""
        predictions = ["C", "C", "G", "G"]
        references = ["C", "N", "G", "N"]
        matches, total, skipped = frame_accuracy(
            predictions, references, ignore_label="N"
        )
        assert matches == 2
        assert total == 2
        assert skipped == 2


class TestFrameRootAccuracy:
    """frame_root_accuracy のテスト"""

    def test_root_match(self):
        """ルート一致（品質は異なる）"""
        predictions = ["C", "G"]
        references = ["C", "G"]
        matches, total, skipped = frame_root_accuracy(predictions, references)
        assert matches == 2
        assert total == 2

    def test_root_mismatch(self):
        """ルート不一致"""
        predictions = ["C", "Am"]
        references = ["G", "G"]
        matches, total, skipped = frame_root_accuracy(predictions, references)
        assert matches == 0
        assert total == 2


class TestEvaluateChordPair:
    """evaluate_chord_pair のテスト"""

    def test_basic_evaluation(self):
        """基本的な評価"""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # テスト用 lab ファイルを作成
            pred_file = tmpdir_path / "test_pred.lab"
            ref_file = tmpdir_path / "test_ref.lab"

            pred_file.write_text("0.0\t1.0\tC\n1.0\t2.0\tG\n")
            ref_file.write_text("0.0\t1.0\tC\n1.0\t2.0\tG\n")

            result = evaluate_chord_pair(
                pred_file, ref_file, frame_rate=10.0, ignore_label=None
            )

            assert isinstance(result, FileMetrics)
            assert isinstance(result.metrics, dict)
            assert isinstance(result.meta, object)  # FileMeta dataclass
            assert isinstance(result.breakdown, object)  # ChordBreakdown dataclass
            assert result.metrics["chord_accuracy"] == 1.0
            assert result.metrics["chord_root_accuracy"] == 1.0
            assert result.breakdown.matches == 20
            assert result.breakdown.total == 20


class TestEvaluateChordDirectory:
    """evaluate_chord_directory のテスト"""

    def test_basic_directory_evaluation(self):
        """基本的なディレクトリ評価"""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pred_dir = tmpdir_path / "predicted"
            ref_dir = tmpdir_path / "reference"
            pred_dir.mkdir()
            ref_dir.mkdir()

            # テスト用ファイルを作成
            for i in range(2):
                pred_file = pred_dir / f"test_{i}.lab"
                ref_file = ref_dir / f"test_{i}.lab"
                pred_file.write_text("0.0\t1.0\tC\n")
                ref_file.write_text("0.0\t1.0\tC\n")

            result = evaluate_chord_directory(
                pred_dir, ref_dir, frame_rate=10.0, use_parallel=False
            )

            assert isinstance(result, DirectoryMetrics)
            assert hasattr(result, "overall")
            assert hasattr(result, "files")
            assert hasattr(result, "missing_predictions")
            assert result.overall["chord_accuracy"] == 1.0
            assert len(result.files) == 2

    def test_missing_predictions(self):
        """見つからないファイルを検出"""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pred_dir = tmpdir_path / "predicted"
            ref_dir = tmpdir_path / "reference"
            pred_dir.mkdir()
            ref_dir.mkdir()

            # 参照ファイルのみ作成
            ref_file = ref_dir / "test.lab"
            ref_file.write_text("0.0\t1.0\tC\n")

            result = evaluate_chord_directory(
                pred_dir, ref_dir, frame_rate=10.0, use_parallel=False
            )

            assert len(result.missing_predictions) == 1
            assert "test.lab" in result.missing_predictions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
