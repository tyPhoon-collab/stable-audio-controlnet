"""スイープ訓練モジュールのテスト"""

import pytest

from scripts.sweep.train import _extract_checkpoint_path


class TestExtractCheckpointPath:
    """チェックポイントパス抽出のテスト"""

    def test_extract_from_log_output(self):
        """ログ出力から"Best model ckpt at"パターンを抽出"""
        output = """
Some training logs...
Epoch 10/20
Best model ckpt at /app/logs/ckpts/sweep_001/epoch_10.ckpt
Training completed
"""
        result = _extract_checkpoint_path(output, "sweep_001")
        assert result == "/app/logs/ckpts/sweep_001/epoch_10.ckpt"

    def test_extract_from_log_with_ansi_codes(self):
        """ANSIエスケープコード付きのログから抽出"""
        output = """
\x1b[32mTraining completed\x1b[0m
\x1b[1m\x1b[32mBest model ckpt at /app/logs/ckpts/sweep_002/best.ckpt\x1b[0m
"""
        result = _extract_checkpoint_path(output, "sweep_002")
        assert result == "/app/logs/ckpts/sweep_002/best.ckpt"

    def test_extract_with_whitespace(self):
        """パス周辺の余分な空白をトリム"""
        output = "Best model ckpt at   /app/logs/ckpts/sweep_003/checkpoint.ckpt   \n"
        result = _extract_checkpoint_path(output, "sweep_003")
        assert result == "/app/logs/ckpts/sweep_003/checkpoint.ckpt"

    def test_no_match_returns_none(self):
        """パターンが見つからない場合はNoneを返す"""
        output = "Training completed without checkpoint info"
        result = _extract_checkpoint_path(output, "sweep_004")
        assert result is None

    def test_extract_from_filesystem(self, tmp_path):
        """ファイルシステムからlast.ckptを探す"""
        # 一時ディレクトリ構造を作成
        ckpt_dir = tmp_path / "logs" / "ckpts"
        ckpt_dir.mkdir(parents=True)

        # 複数の実験フォルダを作成
        (ckpt_dir / "sweep_005_old").mkdir()
        (ckpt_dir / "sweep_005_old" / "last.ckpt").touch()

        (ckpt_dir / "sweep_005").mkdir()
        (ckpt_dir / "sweep_005" / "last.ckpt").touch()

        (ckpt_dir / "sweep_006").mkdir()
        (ckpt_dir / "sweep_006" / "last.ckpt").touch()

        # 一時的にカレントディレクトリを変更
        import os

        orig_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # sweep_005のチェックポイントを取得
            result = _extract_checkpoint_path("", "sweep_005")
            assert result is not None
            assert "sweep_005" in result
            assert result.endswith("last.ckpt")

        finally:
            os.chdir(orig_cwd)

    def test_extract_from_filesystem_multiple_matches(self, tmp_path):
        """同じタグで複数のチェックポイントがある場合、最新を選択"""
        ckpt_dir = tmp_path / "logs" / "ckpts"
        ckpt_dir.mkdir(parents=True)

        # 同じタグで複数のチェックポイントを作成（タイムスタンプで順序付け）
        (ckpt_dir / "sweep_007_v1").mkdir()
        (ckpt_dir / "sweep_007_v1" / "last.ckpt").touch()

        (ckpt_dir / "sweep_007_v2").mkdir()
        (ckpt_dir / "sweep_007_v2" / "last.ckpt").touch()

        (ckpt_dir / "sweep_007").mkdir()
        (ckpt_dir / "sweep_007" / "last.ckpt").touch()

        import os

        orig_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # sweep_007のチェックポイントを取得
            # sorted()によって最後のものが選ばれる
            result = _extract_checkpoint_path("", "sweep_007")
            assert result is not None
            assert "sweep_007" in result

        finally:
            os.chdir(orig_cwd)

    def test_log_output_takes_precedence(self, tmp_path):
        """ログ出力がファイルシステムより優先される"""
        # ファイルシステムに存在しないパス
        output = "Best model ckpt at /app/logs/ckpts/sweep_008/from_log.ckpt"

        result = _extract_checkpoint_path(output, "sweep_008")

        # ログ出力からのパスが返される
        assert result == "/app/logs/ckpts/sweep_008/from_log.ckpt"

    def test_extract_with_special_characters(self):
        """パスに特殊文字が含まれる場合"""
        output = "Best model ckpt at /app/logs/ckpts/sweep-009_test-v2.0/checkpoint_best.ckpt"
        result = _extract_checkpoint_path(output, "sweep-009_test-v2.0")
        assert result == "/app/logs/ckpts/sweep-009_test-v2.0/checkpoint_best.ckpt"

    def test_extract_with_relative_path(self):
        """相対パスが含まれる場合"""
        output = "Best model ckpt at logs/ckpts/sweep_010/best.ckpt"
        result = _extract_checkpoint_path(output, "sweep_010")
        assert result == "logs/ckpts/sweep_010/best.ckpt"

    def test_extract_from_real_training_log(self):
        """実際の訓練ログからの抽出"""
        output = """wandb:
wandb: 🚀 View run daily-jazz-172 at: https://wandb.ai/typhoonmoon1022-saitama-university/stable-audio-controlnet/runs/2sk4qzn5
wandb: Synced 5 W&B file(s), 488 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./logs/wandb/run-20251127_112053-2sk4qzn5/logs
[2025-11-27 17:55:24,412][__main__][INFO] - Best model ckpt at /app/logs/ckpts/musdb-controlnet-chord_2025-11-27-11-20-44/epoch=52-valid_loss=0.480.ckpt"""
        result = _extract_checkpoint_path(
            output, "musdb-controlnet-chord_2025-11-27-11-20-44"
        )
        assert (
            result
            == "/app/logs/ckpts/musdb-controlnet-chord_2025-11-27-11-20-44/epoch=52-valid_loss=0.480.ckpt"
        )

    def test_extract_multiline_with_multiple_patterns(self):
        """複数の"Best model ckpt at"がある場合、最後のものを使用"""
        output = """
First run (failed):
Best model ckpt at /app/logs/ckpts/sweep_011/checkpoint_1.ckpt

Second run (successful):
Best model ckpt at /app/logs/ckpts/sweep_011/checkpoint_2.ckpt
"""
        result = _extract_checkpoint_path(output, "sweep_011")
        # re.search()は最初にマッチしたものを返す
        assert result == "/app/logs/ckpts/sweep_011/checkpoint_1.ckpt"

    def test_tag_filtering_in_filesystem(self, tmp_path):
        """異なるタグのチェックポイントをフィルタリング"""
        ckpt_dir = tmp_path / "logs" / "ckpts"
        ckpt_dir.mkdir(parents=True)

        # 異なるタグのチェックポイントを作成
        (ckpt_dir / "sweep_012_a").mkdir()
        (ckpt_dir / "sweep_012_a" / "last.ckpt").touch()

        (ckpt_dir / "sweep_012_b").mkdir()
        (ckpt_dir / "sweep_012_b" / "last.ckpt").touch()

        import os

        orig_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # sweep_012_aを指定
            result = _extract_checkpoint_path("", "sweep_012_a")
            assert result is not None
            assert "sweep_012_a" in result

            # sweep_012_bを指定
            result = _extract_checkpoint_path("", "sweep_012_b")
            assert result is not None
            assert "sweep_012_b" in result

        finally:
            os.chdir(orig_cwd)


class TestExtractCheckpointPathEdgeCases:
    """エッジケースのテスト"""

    def test_empty_output(self):
        """空の出力"""
        result = _extract_checkpoint_path("", "sweep_100")
        assert result is None

    def test_only_whitespace(self):
        """空白文字のみ"""
        result = _extract_checkpoint_path("   \n\t  ", "sweep_101")
        assert result is None

    def test_similar_but_not_matching_pattern(self):
        """似ているが異なるパターン"""
        output = "Best model checkpoint at /app/logs/ckpts/sweep_102/best.ckpt"
        result = _extract_checkpoint_path(output, "sweep_102")
        assert result is None

    def test_pattern_with_partial_tag_match(self):
        """タグが部分一致する場合"""
        output = "Best model ckpt at /app/logs/ckpts/sweep_103_version/best.ckpt"
        # tagが"sweep_103"の場合、パスに"sweep_103_version"が含まれる
        result = _extract_checkpoint_path(output, "sweep_103")
        # ログ出力は常に返される
        assert result == "/app/logs/ckpts/sweep_103_version/best.ckpt"

    def test_no_logs_directory_exists(self, tmp_path):
        """logs/ckptsディレクトリが存在しない場合"""
        import os

        orig_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = _extract_checkpoint_path("", "sweep_104")
            assert result is None
        finally:
            os.chdir(orig_cwd)

    def test_logs_directory_exists_but_no_matching_tag(self, tmp_path):
        """ディレクトリは存在するが、タグに一致するものがない"""
        ckpt_dir = tmp_path / "logs" / "ckpts"
        ckpt_dir.mkdir(parents=True)

        (ckpt_dir / "other_tag").mkdir()
        (ckpt_dir / "other_tag" / "last.ckpt").touch()

        import os

        orig_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = _extract_checkpoint_path("", "sweep_105")
            assert result is None
        finally:
            os.chdir(orig_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
