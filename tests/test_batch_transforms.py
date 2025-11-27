"""AudioBatchとTransformパイプラインのテスト"""

import pytest
import torch

from main.data.batch import AudioBatch
from main.data.transforms import (
    Compose,
    Identity,
    PerSamplePitchShiftTransform,
    PitchShiftTransform,
    PromptTransform,
    RandomPromptTransform,
    StemGainTransform,
    StemMixTransform,
    VocalsDropTransform,
    create_chord_eval_transform,
    create_chord_training_transform,
)


@pytest.fixture
def sample_stems() -> dict[str, torch.Tensor]:
    """テスト用のステムデータ"""
    return {
        "drums": torch.randn(2, 44100),
        "bass": torch.randn(2, 44100),
        "other": torch.randn(2, 44100),
        "vocals": torch.randn(2, 44100),
    }


@pytest.fixture
def sample_batch(sample_stems) -> AudioBatch:
    """テスト用のAudioBatch"""
    batch_size = 2
    return AudioBatch(
        stems=[sample_stems.copy() for _ in range(batch_size)],
        sample_keys=["song1", "song2"],
        start_seconds=[0.0, 10.0],
        total_seconds=[60.0, 60.0],
        prompts=["", ""],
        chord=torch.randint(-1, 12, (batch_size, 100, 3)),
    )


class TestAudioBatch:
    """AudioBatchクラスのテスト"""

    def test_batch_size(self, sample_batch):
        """バッチサイズの取得"""
        assert sample_batch.batch_size == 2

    def test_get_stem_names(self, sample_batch):
        """ステム名の取得"""
        names = sample_batch.get_stem_names()
        assert names == {"drums", "bass", "other", "vocals"}

    def test_has_audio_false(self, sample_batch):
        """音声がない場合"""
        assert not sample_batch.has_audio()

    def test_has_audio_true(self, sample_batch):
        """音声がある場合"""
        sample_batch.audio = torch.randn(2, 2, 44100)
        assert sample_batch.has_audio()

    def test_to_device(self, sample_batch):
        """デバイス移動"""
        batch = sample_batch.to("cpu")
        assert batch.stems[0]["drums"].device.type == "cpu"
        assert batch.chord.device.type == "cpu"

    def test_validate_success(self, sample_batch):
        """検証成功"""
        sample_batch.validate()  # 例外が発生しないことを確認

    def test_validate_failure_sample_keys(self, sample_batch):
        """sample_keysの長さが不一致の場合"""
        sample_batch.sample_keys = ["song1"]
        with pytest.raises(ValueError, match="sample_keys length mismatch"):
            sample_batch.validate()


class TestVocalsDropTransform:
    """VocalsDropTransformのテスト"""

    def test_drop_vocals(self, sample_batch):
        """ボーカルが除去される"""
        transform = VocalsDropTransform()
        result = transform(sample_batch)

        for stems in result.stems:
            assert "vocals" not in stems
            assert "drums" in stems
            assert "bass" in stems
            assert "other" in stems


class TestStemMixTransform:
    """StemMixTransformのテスト"""

    def test_mix_all(self, sample_batch):
        """全ステムをミックス"""
        transform = StemMixTransform(strategy="all", drop_vocals=True)
        result = transform(sample_batch)

        assert result.audio is not None
        assert result.audio.shape[0] == 2  # batch_size
        assert result.audio.shape[1] == 2  # channels

    def test_mix_random_subset(self, sample_batch):
        """ランダムサブセットをミックス"""
        transform = StemMixTransform(strategy="random_subset", drop_vocals=True)
        result = transform(sample_batch)

        assert result.audio is not None
        assert result.audio.shape[0] == 2


class TestStemGainTransform:
    """StemGainTransformのテスト"""

    def test_gain_applied(self, sample_batch):
        """ゲインが適用される"""
        # 確率1.0で必ず適用
        transform = StemGainTransform(gain_min=2.0, gain_max=2.0, p=1.0)
        original_energy = sample_batch.stems[0]["drums"].abs().sum().item()

        result = transform(sample_batch)
        new_energy = result.stems[0]["drums"].abs().sum().item()

        # ゲイン2.0なのでエネルギーは約2倍
        assert abs(new_energy / original_energy - 2.0) < 0.01

    def test_gain_not_applied(self, sample_batch):
        """確率0.0では適用されない"""
        transform = StemGainTransform(gain_min=2.0, gain_max=2.0, p=0.0)
        original = sample_batch.stems[0]["drums"].clone()

        result = transform(sample_batch)

        assert torch.allclose(result.stems[0]["drums"], original)


class TestPromptTransforms:
    """プロンプト変換のテスト"""

    def test_fixed_prompt(self, sample_batch):
        """固定プロンプト"""
        transform = PromptTransform(prompt="test prompt")
        result = transform(sample_batch)

        assert result.prompts == ["test prompt", "test prompt"]

    def test_random_prompt(self, sample_batch):
        """ランダムプロンプト"""
        choices = ["a", "b", "c"]
        transform = RandomPromptTransform(choices=choices)
        result = transform(sample_batch)

        for prompt in result.prompts:
            assert prompt in choices


class TestPitchShiftTransform:
    """PitchShiftTransformのテスト"""

    def test_chord_shift(self):
        """和音テンソルのシフト"""
        transform = PitchShiftTransform(
            semitones_min=1, semitones_max=1, sample_rate=44100, p=1.0
        )

        # 和音テンソル: root=0 (C), quality=0, inversion=0
        chord = torch.tensor([[[0, 0, 0], [5, 1, 0], [-1, -1, 0]]])

        batch = AudioBatch(
            stems=[{"drums": torch.randn(2, 44100)}],
            sample_keys=["song1"],
            chord=chord,
        )

        result = transform(batch)

        # root=0 -> root=1, root=5 -> root=6, root=-1 -> root=-1 (変化なし)
        assert result.chord is not None
        assert result.chord[0, 0, 0].item() == 1
        assert result.chord[0, 1, 0].item() == 6
        assert result.chord[0, 2, 0].item() == -1

    def test_chord_shift_wrap_around(self):
        """和音テンソルのラップアラウンド"""
        transform = PitchShiftTransform(
            semitones_min=2, semitones_max=2, sample_rate=44100, p=1.0
        )

        # root=11 (B) + 2 = 1 (C#)
        chord = torch.tensor([[[11, 0, 0]]])

        batch = AudioBatch(
            stems=[{"drums": torch.randn(2, 44100)}],
            sample_keys=["song1"],
            chord=chord,
        )

        result = transform(batch)

        assert result.chord is not None
        assert result.chord[0, 0, 0].item() == 1  # (11 + 2) % 12 = 1


class TestPerSamplePitchShiftTransform:
    """PerSamplePitchShiftTransformのテスト"""

    def test_per_sample_shift(self):
        """サンプルごとに異なるシフト"""
        transform = PerSamplePitchShiftTransform(
            semitones_min=-5, semitones_max=6, sample_rate=44100, p=1.0
        )

        batch = AudioBatch(
            stems=[
                {"drums": torch.randn(2, 44100)},
                {"drums": torch.randn(2, 44100)},
            ],
            sample_keys=["song1", "song2"],
            chord=torch.randint(0, 12, (2, 100, 3)),
        )

        result = transform(batch)

        # 変換後も形状は保持される
        assert len(result.stems) == 2
        assert result.chord is not None
        assert result.chord.shape == (2, 100, 3)


class TestCompose:
    """Composeのテスト"""

    def test_compose_multiple(self, sample_batch):
        """複数のTransformを順次適用"""
        transform = Compose(
            [
                VocalsDropTransform(),
                StemMixTransform(strategy="all", drop_vocals=False),
                PromptTransform(prompt="composed"),
            ]
        )

        result = transform(sample_batch)

        # ボーカルが除去されている
        for stems in result.stems:
            assert "vocals" not in stems

        # 音声がミックスされている
        assert result.audio is not None

        # プロンプトが設定されている
        assert result.prompts == ["composed", "composed"]


class TestIdentity:
    """Identityのテスト"""

    def test_identity(self, sample_batch):
        """何も変更しない"""
        transform = Identity()
        result = transform(sample_batch)

        # 同じオブジェクトが返される
        assert result is sample_batch


class TestFactoryFunctions:
    """ファクトリ関数のテスト"""

    def test_create_chord_training_transform(self):
        """訓練用Transform作成"""
        transform = create_chord_training_transform(
            csv_path=None,
            drop_vocals=True,
            pitch_shift_p=0.5,
            gain_augment_p=0.5,
            sample_rate=44100,
        )

        assert isinstance(transform, Compose)

    def test_create_chord_eval_transform(self):
        """評価用Transform作成"""
        transform = create_chord_eval_transform(
            csv_path=None,
            drop_vocals=True,
        )

        assert isinstance(transform, Compose)
