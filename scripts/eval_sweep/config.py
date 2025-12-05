"""評価パラメータスイープの設定管理"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class EvalExperiment:
    """評価実験"""

    name: str
    cfg_scale: float
    steps: int
    status: str = "pending"  # pending, evaluated, failed
    metrics: dict = field(default_factory=dict)
    error_message: str = ""


@dataclass
class EvalSweepConfig:
    """評価パラメータスイープ設定"""

    checkpoint_path: str
    exp_config: str
    output_dir: Path
    experiments: list[EvalExperiment]
    samples: int = 50
    batch_size: int = 16
    dry_run: bool = False

    # 和音評価用の固定パラメータ
    chord_frame_rate: int = 10
    clap_model_path: str = "ckpts/music_audioset_epoch_15_esc_90.14.pt"

    # Hydraオーバーライド（バックボーン設定など）
    hydra_overrides: list[str] = field(default_factory=list)

    _state_file: Path = field(default=None, repr=False)

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self._state_file = self.output_dir / "eval_sweep_state.json"

    def save_state(self) -> None:
        """状態をファイルに保存"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "checkpoint_path": self.checkpoint_path,
            "exp_config": self.exp_config,
            "samples": self.samples,
            "batch_size": self.batch_size,
            "hydra_overrides": self.hydra_overrides,
            "experiments": [
                {
                    "name": e.name,
                    "cfg_scale": e.cfg_scale,
                    "steps": e.steps,
                    "status": e.status,
                    "metrics": e.metrics,
                    "error_message": e.error_message,
                }
                for e in self.experiments
            ],
        }
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_state(self) -> bool:
        """状態をファイルから読み込み"""
        if not self._state_file.exists():
            return False

        with open(self._state_file) as f:
            state = json.load(f)

        # 実験の状態を復元
        state_map = {e["name"]: e for e in state.get("experiments", [])}
        for exp in self.experiments:
            if exp.name in state_map:
                saved = state_map[exp.name]
                exp.status = saved.get("status", "pending")
                exp.metrics = saved.get("metrics", {})
                exp.error_message = saved.get("error_message", "")

        return True


def load_eval_sweep_config(
    config_path: str,
    *,
    output_dir: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
    checkpoint_path: str | None = None,
    hydra_overrides: list[str] | None = None,
) -> EvalSweepConfig:
    """YAML設定ファイルから評価スイープ設定を読み込み"""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # チェックポイントパス（CLIオーバーライド優先）
    ckpt = checkpoint_path or raw.get("checkpoint")
    if not ckpt:
        raise ValueError(
            "チェックポイントパスが指定されていません（--ckpt または設定ファイルのcheckpoint）"
        )

    # 出力ディレクトリ
    if output_dir:
        out_dir = Path(output_dir)
    else:
        config_name = Path(config_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("out") / "eval_sweep" / config_name / timestamp

    # スイープパラメータ
    sweep_params = raw.get("sweep_params", {})
    cfg_scales = sweep_params.get("cfg_scale", [7.0])
    steps_list = sweep_params.get("steps", [100])

    # 固定パラメータ
    eval_params = raw.get("eval_params", {})
    samples = eval_params.get("samples", 50)
    batch_size = eval_params.get("batch_size", 16)

    # 実験を生成
    experiments = []
    for cfg in cfg_scales:
        for steps in steps_list:
            name = f"cfg{cfg}_steps{steps}"
            experiments.append(
                EvalExperiment(
                    name=name,
                    cfg_scale=float(cfg),
                    steps=int(steps),
                )
            )

    # Hydraオーバーライド
    overrides = hydra_overrides or []
    if "hydra_overrides" in raw:
        overrides = raw["hydra_overrides"] + overrides

    config = EvalSweepConfig(
        checkpoint_path=ckpt,
        exp_config=raw.get("base_exp", "train_musdb_controlnet_chord"),
        output_dir=out_dir,
        experiments=experiments,
        samples=samples,
        batch_size=batch_size,
        dry_run=dry_run,
        hydra_overrides=overrides,
    )

    # 状態復元
    if resume:
        config.load_state()

    # 初期状態を保存
    config.save_state()

    return config
