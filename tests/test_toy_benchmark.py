"""Toyベンチマークモジュールのテスト"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from main.eval.toy_benchmark import (
    ChordTestResult,
    ToyBenchmarkResult,
    create_progression_lab,
    create_single_chord_lab,
    evaluate_against_expected,
    evaluate_progression_test,
    evaluate_single_chord_test,
    load_toy_test_config,
)


class TestCreateSingleChordLab:
    """create_single_chord_lab のテスト"""

    def test_basic(self):
        """基本的な単一コードlab作成"""
        result = create_single_chord_lab("C:maj", 10.0)
        assert len(result) == 1
        assert result[0] == (0.0, 10.0, "C:maj")

    def test_minor_chord(self):
        """マイナーコード"""
        result = create_single_chord_lab("A:min", 5.0)
        assert result[0] == (0.0, 5.0, "A:min")


class TestCreateProgressionLab:
    """create_progression_lab のテスト"""

    def test_four_chords(self):
        """4コードの進行"""
        result = create_progression_lab("C:maj,G:maj,A:min,F:maj", 16.0)
        assert len(result) == 4
        assert result[0] == (0.0, 4.0, "C:maj")
        assert result[1] == (4.0, 8.0, "G:maj")
        assert result[2] == (8.0, 12.0, "A:min")
        assert result[3] == (12.0, 16.0, "F:maj")

    def test_two_chords(self):
        """2コードの進行"""
        result = create_progression_lab("C:maj,G:maj", 10.0)
        assert len(result) == 2
        assert result[0] == (0.0, 5.0, "C:maj")
        assert result[1] == (5.0, 10.0, "G:maj")

    def test_empty_progression(self):
        """空の進行"""
        result = create_progression_lab("", 10.0)
        assert len(result) == 0


class TestEvaluateAgainstExpected:
    """evaluate_against_expected のテスト"""

    def test_perfect_match(self):
        """完全一致"""
        with TemporaryDirectory() as tmpdir:
            # 予測labファイルを作成
            pred_file = Path(tmpdir) / "pred.lab"
            pred_file.write_text("0.0\t10.0\tC:maj\n")

            # 期待されるアノテーション
            expected = [(0.0, 10.0, "C:maj")]

            result = evaluate_against_expected(pred_file, expected, frame_rate=10.0)

            assert result.accuracy == 1.0
            assert result.root_accuracy == 1.0
            assert result.matches == result.total

    def test_partial_match(self):
        """部分一致"""
        with TemporaryDirectory() as tmpdir:
            # 予測labファイルを作成（半分だけ一致）
            pred_file = Path(tmpdir) / "pred.lab"
            pred_file.write_text("0.0\t5.0\tC:maj\n5.0\t10.0\tG:maj\n")

            # 期待されるアノテーション（全部C:maj）
            expected = [(0.0, 10.0, "C:maj")]

            result = evaluate_against_expected(pred_file, expected, frame_rate=10.0)

            # 半分が一致するはず
            assert 0.4 <= result.accuracy <= 0.6
            assert result.total > 0

    def test_root_match_quality_mismatch(self):
        """根音は一致するが品質が異なる"""
        with TemporaryDirectory() as tmpdir:
            pred_file = Path(tmpdir) / "pred.lab"
            pred_file.write_text("0.0\t10.0\tC:min\n")  # C:minと予測

            expected = [(0.0, 10.0, "C:maj")]  # C:majが正解

            result = evaluate_against_expected(pred_file, expected, frame_rate=10.0)

            # 根音は一致するが完全一致しない
            assert result.accuracy == 0.0  # 完全一致なし
            assert result.root_accuracy == 1.0  # 根音は一致


class TestEvaluateSingleChordTest:
    """evaluate_single_chord_test のテスト"""

    def test_perfect_match(self):
        """完全一致の単一コードテスト"""
        with TemporaryDirectory() as tmpdir:
            pred_file = Path(tmpdir) / "C_maj_test.lab"
            pred_file.write_text("0.0\t10.0\tC:maj\n")

            result = evaluate_single_chord_test(
                predicted_lab_path=pred_file,
                chord="C:maj",
                duration=10.0,
                frame_rate=10.0,
            )

            assert result.test_name == "single_chord_C:maj"
            assert result.chord_or_progression == "C:maj"
            assert result.accuracy == 1.0


class TestEvaluateProgressionTest:
    """evaluate_progression_test のテスト"""

    def test_perfect_match(self):
        """完全一致のコード進行テスト"""
        with TemporaryDirectory() as tmpdir:
            pred_file = Path(tmpdir) / "pop_punk_test.lab"
            pred_file.write_text(
                "0.0\t4.0\tC:maj\n4.0\t8.0\tG:maj\n8.0\t12.0\tA:min\n12.0\t16.0\tF:maj\n"
            )

            result = evaluate_progression_test(
                predicted_lab_path=pred_file,
                progression_name="pop_punk",
                progression="C:maj,G:maj,A:min,F:maj",
                duration=16.0,
                frame_rate=10.0,
            )

            assert result.test_name == "progression_pop_punk"
            assert result.chord_or_progression == "C:maj,G:maj,A:min,F:maj"
            assert result.accuracy == 1.0


class TestToyBenchmarkResult:
    """ToyBenchmarkResult のテスト"""

    def test_empty_result(self):
        """空の結果"""
        result = ToyBenchmarkResult()
        result_dict = result.to_dict()

        assert result_dict["summary"]["single_chord"]["count"] == 0
        assert result_dict["summary"]["progression"]["count"] == 0
        assert result_dict["summary"]["prompt"]["count"] == 0

    def test_with_results(self):
        """結果ありの場合"""
        result = ToyBenchmarkResult()
        result.single_chord_results.append(
            ChordTestResult(
                test_name="single_chord_C:maj",
                chord_or_progression="C:maj",
                accuracy=0.8,
                root_accuracy=0.9,
                matches=80,
                total=100,
                skipped=0,
                root_matches=90,
            )
        )

        result_dict = result.to_dict()

        assert result_dict["summary"]["single_chord"]["count"] == 1
        assert result_dict["summary"]["single_chord"]["avg_accuracy"] == 0.8
        assert result_dict["summary"]["single_chord"]["avg_root_accuracy"] == 0.9


class TestLoadToyTestConfig:
    """load_toy_test_config のテスト"""

    def test_load_default_config(self):
        """デフォルト設定ファイルの読み込み"""
        config_path = Path("data/toy_test_config.json")
        if config_path.exists():
            config = load_toy_test_config(config_path)

            assert "single_chords" in config
            assert "common_progressions" in config
            assert "chord_friendly_prompts" in config
            assert "evaluation" in config

    def test_file_not_found(self):
        """設定ファイルが見つからない場合"""
        with pytest.raises(FileNotFoundError):
            load_toy_test_config("nonexistent_config.json")
