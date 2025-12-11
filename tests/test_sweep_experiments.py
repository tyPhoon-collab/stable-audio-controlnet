"""スイープ実験スクリプトのテスト"""

import tempfile
from pathlib import Path

import yaml

from scripts.sweep_experiments import (
    ExperimentConfig,
    SweepState,
    generate_experiments,
    load_state,
    load_sweep_config,
    save_state,
)


class TestGenerateExperiments:
    """generate_experiments のテスト"""

    def test_single_param(self):
        """単一パラメータのスイープ"""
        config = {
            "base_exp": "test_exp",
            "base_tag": "test",
            "parameters": {
                "model.lr": [1e-4, 1e-5],
            },
            "eval_params": {
                "cfg_scale": [7.0],
                "steps": [100],
            },
        }

        experiments = generate_experiments(config)

        assert len(experiments) == 2
        assert experiments[0].hydra_overrides == ["model.lr=0.0001"]
        assert experiments[1].hydra_overrides == ["model.lr=1e-05"]

    def test_multiple_params(self):
        """複数パラメータの組み合わせ"""
        config = {
            "base_exp": "test_exp",
            "base_tag": "test",
            "parameters": {
                "model.lr": [1e-4, 1e-5],
                "model.depth_factor": [0.5, 0.7],
            },
        }

        experiments = generate_experiments(config)

        # 2 x 2 = 4 実験
        assert len(experiments) == 4

    def test_empty_params(self):
        """パラメータなしの場合"""
        config = {
            "base_exp": "test_exp",
            "base_tag": "test",
            "parameters": {},
        }

        experiments = generate_experiments(config)

        assert len(experiments) == 1
        assert experiments[0].hydra_overrides == []

    def test_wandb_sweep_format(self):
        """W&B sweep形式のパラメータ"""
        config = {
            "base_exp": "test_exp",
            "base_tag": "test",
            "parameters": {
                "model.loss_type": {"values": ["mse", "l1"]},
                "model.lr": {"value": 1e-5},  # 単一値
            },
        }

        experiments = generate_experiments(config)

        # loss_type: 2, lr: 1 => 2実験
        assert len(experiments) == 2


class TestStatePersistence:
    """状態の保存・読み込みテスト"""

    def test_save_and_load_state(self):
        """状態の保存と読み込み"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"

            experiments = [
                ExperimentConfig(
                    name="test_exp_001",
                    tag="test_001",
                    exp_config="train_musdb_controlnet_chord",
                    hydra_overrides=["model.lr=1e-5"],
                    eval_params={"cfg_scale": [7.0], "steps": [100]},
                    status="completed",
                    checkpoint_path="/path/to/ckpt",
                    metrics={"accuracy": 0.85},
                ),
            ]

            state = SweepState(
                config_path="/path/to/config.yaml",
                output_dir=tmpdir,
                experiments=experiments,
                current_index=0,
            )

            save_state(state, state_path)

            # 読み込み
            loaded = load_state(state_path)

            assert loaded.config_path == state.config_path
            assert len(loaded.experiments) == 1
            assert loaded.experiments[0].name == "test_exp_001"
            assert loaded.experiments[0].status == "completed"
            assert loaded.experiments[0].metrics == {"accuracy": 0.85}


class TestLoadSweepConfig:
    """設定ファイル読み込みテスト"""

    def test_load_yaml_config(self):
        """YAML設定ファイルの読み込み"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "base_exp": "test_exp",
                "base_tag": "test",
                "parameters": {
                    "model.lr": [1e-4, 1e-5],
                },
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            loaded = load_sweep_config(str(config_path))

            assert loaded["base_exp"] == "test_exp"
            assert loaded["parameters"]["model.lr"] == [1e-4, 1e-5]


class TestExperimentConfig:
    """ExperimentConfig のテスト"""

    def test_default_values(self):
        """デフォルト値の確認"""
        exp = ExperimentConfig(
            name="test",
            tag="test",
            exp_config="test_exp",
            hydra_overrides=[],
            eval_params={},
        )

        assert exp.status == "pending"
        assert exp.checkpoint_path is None
        assert exp.metrics == {}


class TestExperimentsFormat:
    """experiments形式のテスト"""

    def test_experiments_config_generation(self):
        """experiments形式から正しく実験が生成されるか"""
        config = {
            "base_exp": "train_musdb_controlnet_chord",
            "base_tag": "sweep-backbone",
            "experiments": [
                {
                    "name": "mamba",
                    "params": {
                        "model.chord_conditioner.backbone._target_": "main.chord_backbones.ChordMambaBackbone",
                    },
                },
                {
                    "name": "gru",
                    "params": {
                        "model.chord_conditioner.backbone._target_": "main.chord_backbones.ChordGRUBackbone",
                    },
                },
            ],
        }

        experiments = generate_experiments(config)

        assert len(experiments) == 2
        assert "mamba" in experiments[0].name
        assert "gru" in experiments[1].name
