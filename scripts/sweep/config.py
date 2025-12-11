"""スイープ実験の設定管理

オプションを最小限に整理し、デフォルト値で動作するように設計
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalConfig:
    """評価設定（シンプル化）"""

    samples: int = 100
    batch_size: int = 32
    cfg_scale: float = 7.0
    steps: int = 100
    chord_dict: str = "submission"

    # 固定値（通常変更不要）
    chord_frame_rate: float = 21.533203125
    clap_model_path: str = "ckpts/music_audioset_epoch_15_esc_90.14.pt"


@dataclass
class Experiment:
    """単一実験の設定と状態"""

    name: str
    tag: str
    exp_config: str
    hydra_overrides: list[str] = field(default_factory=list)

    # 状態
    status: str = "pending"  # pending, trained, evaluated, failed
    checkpoint_path: str | None = None
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tag": self.tag,
            "exp_config": self.exp_config,
            "hydra_overrides": self.hydra_overrides,
            "status": self.status,
            "checkpoint_path": self.checkpoint_path,
            "error_message": self.error_message,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experiment:
        return cls(
            name=data["name"],
            tag=data["tag"],
            exp_config=data["exp_config"],
            hydra_overrides=data.get("hydra_overrides", []),
            status=data.get("status", "pending"),
            checkpoint_path=data.get("checkpoint_path"),
            error_message=data.get("error_message"),
            metrics=data.get("metrics", {}),
        )


@dataclass
class SweepConfig:
    """スイープ全体の設定"""

    config_path: Path
    output_dir: Path
    eval_config: EvalConfig
    experiments: list[Experiment]

    # 実行オプション
    resume: bool = False
    train_only: bool = False
    eval_only: bool = False
    dry_run: bool = False

    @property
    def state_file(self) -> Path:
        return self.output_dir / "sweep_state.json"

    def save_state(self) -> None:
        """状態をJSONに保存"""
        import json

        self.output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "config_path": str(self.config_path),
            "output_dir": str(self.output_dir),
            "experiments": [exp.to_dict() for exp in self.experiments],
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_state(self) -> bool:
        """保存された状態を読み込む"""
        import json

        if not self.state_file.exists():
            return False

        with open(self.state_file) as f:
            state = json.load(f)

        # 実験の状態を更新
        saved_experiments = {e["name"]: e for e in state.get("experiments", [])}
        for exp in self.experiments:
            if exp.name in saved_experiments:
                saved = saved_experiments[exp.name]
                exp.status = saved.get("status", exp.status)
                exp.checkpoint_path = saved.get("checkpoint_path")
                exp.metrics = saved.get("metrics", {})
                exp.error_message = saved.get("error_message")

        return True


def load_sweep_config(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    resume: bool = False,
    train_only: bool = False,
    eval_only: bool = False,
    dry_run: bool = False,
    # 評価オプション（オーバーライド用）
    samples: int | None = None,
    batch_size: int | None = None,
    cfg_scale: float | None = None,
    steps: int | None = None,
) -> SweepConfig:
    """設定ファイルからSweepConfigを生成"""
    config_path = Path(config_path)

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # 出力ディレクトリ
    if output_dir is None:
        output_dir = Path("out/sweep") / config_path.stem
    else:
        output_dir = Path(output_dir)

    # 評価設定（ファイル設定 → CLI引数でオーバーライド）
    eval_params = raw_config.get("eval_params", {})

    # リスト形式の場合は最初の値を使用
    def get_scalar(value, default):
        if isinstance(value, list):
            return value[0] if value else default
        return value if value is not None else default

    eval_config = EvalConfig(
        samples=samples
        if samples is not None
        else get_scalar(eval_params.get("samples"), 100),
        batch_size=batch_size
        if batch_size is not None
        else get_scalar(eval_params.get("batch_size"), 32),
        cfg_scale=cfg_scale
        if cfg_scale is not None
        else get_scalar(eval_params.get("cfg_scale"), 7.0),
        steps=steps if steps is not None else get_scalar(eval_params.get("steps"), 100),
        chord_dict=get_scalar(eval_params.get("chord_dict"), "submission"),
    )

    # 実験リストを生成
    experiments = _generate_experiments(raw_config)

    config = SweepConfig(
        config_path=config_path,
        output_dir=output_dir,
        eval_config=eval_config,
        experiments=experiments,
        resume=resume,
        train_only=train_only,
        eval_only=eval_only,
        dry_run=dry_run,
    )

    # 再開モードなら状態を読み込む
    if resume:
        config.load_state()

    return config


def _parse_parameters_section(parameters: dict[str, Any]) -> dict[str, list[Any]]:
    """W&B sweep形式のparametersセクションをパース

    サポートする形式:
    1. values形式: {"param": {"values": [1, 2, 3]}}
    2. value形式（単一値）: {"param": {"value": 1}}
    3. 直接リスト形式: {"param": [1, 2, 3]}
    4. 直接値形式: {"param": 1}
    """
    result = {}
    for key, val in parameters.items():
        if isinstance(val, dict):
            if "values" in val:
                result[key] = val["values"]
            elif "value" in val:
                result[key] = [val["value"]]
            else:
                result[key] = [val]
        elif isinstance(val, list):
            result[key] = val
        else:
            result[key] = [val]
    return result


def _generate_experiments(raw_config: dict[str, Any]) -> list[Experiment]:
    """設定から実験リストを生成

    サポートする設定形式:
    1. experiments: 個別実験定義（バックボーン比較等）
    2. parameters: W&B sweep形式のグリッドサーチ
    """
    experiments = []

    base_exp = raw_config.get("base_exp", "train_musdb_controlnet_chord")
    base_tag = raw_config.get("base_tag", "sweep")

    # experiments形式（個別実験定義）
    explicit_experiments = raw_config.get("experiments", [])
    if explicit_experiments:
        for i, exp_config in enumerate(explicit_experiments):
            exp_name = exp_config.get("name", f"exp_{i}")
            params = exp_config.get("params", {})

            name = f"{base_tag}_{i:03d}_{exp_name}"
            tag = f"{base_tag}_{i:03d}"

            hydra_overrides = _build_hydra_overrides(params)

            experiments.append(
                Experiment(
                    name=name,
                    tag=tag,
                    exp_config=base_exp,
                    hydra_overrides=hydra_overrides,
                )
            )
        return experiments

    # parameters形式（グリッドサーチ）
    raw_params = raw_config.get("parameters", {})
    if raw_params:
        from itertools import product

        train_params = _parse_parameters_section(raw_params)
        param_names = list(train_params.keys())
        param_values = [
            v if isinstance(v, list) else [v] for v in train_params.values()
        ]
        param_combinations = [
            dict(zip(param_names, combo)) for combo in product(*param_values)
        ]

        for i, params in enumerate(param_combinations):
            param_str = "_".join(f"{k}={v}" for k, v in params.items())
            name = f"{base_tag}_{i:03d}_{param_str}"
            tag = f"{base_tag}_{i:03d}"

            hydra_overrides = _build_hydra_overrides(params)

            experiments.append(
                Experiment(
                    name=name,
                    tag=tag,
                    exp_config=base_exp,
                    hydra_overrides=hydra_overrides,
                )
            )
        return experiments

    # デフォルト: 単一実験
    experiments.append(
        Experiment(
            name=f"{base_tag}_000_default",
            tag=f"{base_tag}_000",
            exp_config=base_exp,
        )
    )
    return experiments


def _build_hydra_overrides(params: dict[str, Any]) -> list[str]:
    """パラメータからHydraオーバーライドを生成"""
    overrides = []

    backbone_params = {}
    other_params = {}

    for key, value in params.items():
        hydra_key = key.replace("/", ".")
        if "backbone." in hydra_key or hydra_key.endswith("backbone"):
            backbone_params[hydra_key] = value
        else:
            other_params[hydra_key] = value

    # バックボーンは削除してから追加
    if backbone_params:
        overrides.append("~model.chord_conditioner.backbone")
        for key, value in backbone_params.items():
            overrides.append(f"+{key}={value}")

    for key, value in other_params.items():
        overrides.append(f"{key}={value}")

    return overrides
