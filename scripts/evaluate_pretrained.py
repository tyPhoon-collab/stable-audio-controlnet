#!/usr/bin/env python
"""
事前学習モデル（テキストプロンプトのみ）評価スクリプト

FAD、CLAP、および和音評価（オプション）を行います。

和音評価を行う場合は、事前にISMIR2019で和音推定を実行し、
推定された.labファイルのディレクトリを指定してください。

使用例:
    # FADとCLAPのみ
    python scripts/evaluate_pretrained.py out/pretrained_baseline data/mixtures

    # 和音評価も含める（事前に和音推定が必要）
    python scripts/evaluate_pretrained.py out/pretrained_baseline data/mixtures \
        --predicted-chord-dir out/pretrained_baseline/predicted_chords \
        --reference-chord-dir data/musdb_test

    # 結果をJSONに出力
    python scripts/evaluate_pretrained.py out/pretrained_baseline data/mixtures \
        --output out/pretrained_evaluation_results.json

和音推定の実行方法（別コンテナで実行）:
    cd ISMIR2019-Large-Vocabulary-Chord-Recognition
    python batch_chord_recognition.py \
        --input_dir ../out/pretrained_baseline \
        --output_dir ../out/pretrained_baseline/predicted_chords
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.eval.clap import evaluate_clap_batch, initialize_clap_model
from main.eval.fad import compute_fad_score
from main.eval.metrics import (
    DirectoryMetrics,
    FileMetrics,
    evaluate_chord_directory,
    evaluate_chord_pair,
)


def build_track_name_mapping(
    generated_audio_dir: Path | str,
) -> dict[str, str]:
    """生成されたファイルのJSONからtrack_nameマッピングを構築

    Args:
        generated_audio_dir: 生成された音声ファイルのディレクトリ

    Returns:
        {生成ファイル名（拡張子なし）: track_name} のマッピング
    """
    audio_dir = Path(generated_audio_dir)
    mapping: dict[str, str] = {}

    for json_file in audio_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                track_name = metadata.get("track_name")
                if track_name:
                    # JSONファイル名から拡張子を除いた部分をキーにする
                    file_stem = json_file.stem
                    mapping[file_stem] = track_name
        except Exception:
            continue

    return mapping


def evaluate_chord_with_mapping(
    predicted_chord_dir: Path | str,
    reference_chord_dir: Path | str,
    track_mapping: dict[str, str],
    frame_rate: float,
    ignore_label: str | None = None,
) -> DirectoryMetrics:
    """track_nameマッピングを使用して和音評価を実行

    Args:
        predicted_chord_dir: 推定された和音.labファイルのディレクトリ
        reference_chord_dir: 参照和音.labファイルのディレクトリ
        track_mapping: {予測ファイル名（拡張子なし）: track_name} のマッピング
        frame_rate: フレームレート
        ignore_label: 無視する和音ラベル

    Returns:
        評価結果
    """
    pred_path = Path(predicted_chord_dir)
    ref_path = Path(reference_chord_dir)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference directory not found: {ref_path}")

    per_file: dict[str, FileMetrics] = {}
    missing_predictions: list[str] = []
    evaluated_count = 0

    # 予測ファイルを処理
    for pred_lab in sorted(pred_path.glob("*.lab")):
        pred_stem = pred_lab.stem

        # マッピングからtrack_nameを取得
        track_name = track_mapping.get(pred_stem)
        if not track_name:
            print(f"  警告: {pred_stem} のtrack_nameが見つかりません")
            continue

        # 参照ファイルを探す
        ref_lab = ref_path / f"{track_name}.lab"
        if not ref_lab.exists():
            missing_predictions.append(f"{track_name}.lab (from {pred_stem})")
            continue

        # 評価を実行
        try:
            file_metrics = evaluate_chord_pair(
                pred_lab, ref_lab, frame_rate, ignore_label
            )
            per_file[pred_lab.name] = file_metrics
            evaluated_count += 1
        except Exception as e:
            print(f"  エラー: {pred_lab.name} の評価中: {e}")

    print(f"  評価完了: {evaluated_count} ファイル")

    # 全体的な指標を集計
    total_matches = 0
    total_root_matches = 0
    total_frames = 0
    total_skipped = 0

    for file_metrics in per_file.values():
        breakdown = file_metrics.breakdown
        total_matches += breakdown.matches
        total_root_matches += breakdown.root_matches
        total_frames += breakdown.total
        total_skipped += breakdown.skipped

    overall_chord_accuracy = total_matches / total_frames if total_frames > 0 else 0.0
    overall_chord_root_accuracy = (
        total_root_matches / total_frames if total_frames > 0 else 0.0
    )

    return DirectoryMetrics(
        overall={
            "chord_accuracy": overall_chord_accuracy,
            "chord_root_accuracy": overall_chord_root_accuracy,
        },
        files=per_file,
        missing_predictions=missing_predictions,
    )


def evaluate_clap_with_model(
    model,
    audio_dir: Path | str,
    device: str = "cuda",
    num_workers: int = 4,
) -> dict[str, float | dict[str, float]]:
    """事前ロード済みモデルでCLAPスコアを計算"""
    audio_dir = Path(audio_dir)
    audio_files = sorted(audio_dir.glob("*.wav")) + sorted(audio_dir.glob("*.mp3"))

    if not audio_files:
        return {"overall_clap_score": 0.0, "files": {}}

    # 音声ファイルとプロンプトのペアを収集
    valid_audio_paths: list[Path] = []
    valid_prompts: list[str] = []

    for audio_file in audio_files:
        json_file = audio_file.with_suffix(".json")

        if not json_file.exists():
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                prompt = metadata.get("prompt")

            if prompt:
                valid_audio_paths.append(audio_file)
                valid_prompts.append(prompt)

        except Exception:
            continue

    if not valid_audio_paths:
        return {"overall_clap_score": 0.0, "files": {}}

    # バッチでCLAPスコアを計算
    file_scores = evaluate_clap_batch(
        model=model,
        audio_paths=valid_audio_paths,
        prompts=valid_prompts,
        device=device,
        num_workers=num_workers,
    )

    if not file_scores:
        return {"overall_clap_score": 0.0, "files": {}}

    overall_score = sum(file_scores.values()) / len(file_scores)

    return {
        "overall_clap_score": overall_score,
        "files": file_scores,
    }


def evaluate_pretrained(
    generated_audio_dir: Path | str,
    reference_audio_dir: Path | str,
    predicted_chord_dir: Path | str | None = None,
    reference_chord_dir: Path | str | None = None,
    chord_frame_rate: float = 24.0,
    chord_ignore_label: str | None = None,
    fad_model_name: str = "vggish",
    clap_model_path: str = "ckpts/music_audioset_epoch_15_esc_90.14.pt",
    num_workers: int = 4,
) -> dict:
    """事前学習モデルの評価（FAD、CLAP、およびオプションで和音評価）

    Args:
        generated_audio_dir: 生成された音声ファイルのディレクトリ
        reference_audio_dir: 参照音声ファイルのディレクトリ
        predicted_chord_dir: 推定された和音.labファイルのディレクトリ（オプション）
        reference_chord_dir: 参照和音.labファイルのディレクトリ（オプション）
        chord_frame_rate: 和音評価のフレームレート
        chord_ignore_label: 無視する和音ラベル
        fad_model_name: FAD計算に使用するモデル名
        clap_model_path: CLAPモデルの重みパス
        num_workers: 並列ワーカー数

    Returns:
        評価結果の辞書
    """
    print(f"評価開始: {generated_audio_dir}")
    print(f"参照音声: {reference_audio_dir}")

    # 和音評価の有無を確認
    do_chord_eval = (
        predicted_chord_dir is not None
        and reference_chord_dir is not None
        and Path(predicted_chord_dir).exists()
        and Path(reference_chord_dir).exists()
    )

    if do_chord_eval:
        print("和音評価: 有効")
        print(f"  推定和音: {predicted_chord_dir}")
        print(f"  参照和音: {reference_chord_dir}")
    else:
        print("和音評価: 無効（推定和音ディレクトリが指定されていないか存在しません）")

    # CLAPモデルを事前ロード
    clap_model = None
    try:
        clap_model = initialize_clap_model(weights_path=clap_model_path, device="cuda")
        print("CLAPモデルの読み込み完了")
    except Exception as e:
        print(f"Failed to initialize CLAP model: {e}")

    fad_score = 0.0
    clap_result: dict = {"overall_clap_score": 0.0, "files": {}}
    chord_result = None

    # track_nameマッピングを構築（和音評価用）
    track_mapping: dict[str, str] = {}
    if do_chord_eval:
        print("track_nameマッピングを構築中...")
        track_mapping = build_track_name_mapping(generated_audio_dir)
        print(f"  マッピング数: {len(track_mapping)}")
        if len(track_mapping) == 0:
            print(
                "  警告: マッピングが見つかりません。JSONファイルにtrack_nameが含まれていない可能性があります。"
            )
            print(
                "  eval_pretrained.pyを再実行してtrack_nameを含むJSONを生成してください。"
            )

    def run_fad_eval():
        return compute_fad_score(
            reference_dir=str(reference_audio_dir),
            generated_dir=str(generated_audio_dir),
            model_name=fad_model_name,
            verbose=True,
        )

    def run_clap_eval():
        if clap_model is None:
            return {"overall_clap_score": 0.0, "files": {}}
        return evaluate_clap_with_model(
            model=clap_model,
            audio_dir=generated_audio_dir,
            device="cuda",
            num_workers=num_workers,
        )

    def run_chord_eval():
        if not do_chord_eval:
            return None
        if len(track_mapping) == 0:
            # マッピングがない場合は従来のファイル名ベースで評価
            return evaluate_chord_directory(
                predicted_dir=predicted_chord_dir,
                reference_dir=reference_chord_dir,
                frame_rate=chord_frame_rate,
                ignore_label=chord_ignore_label,
                use_parallel=True,
                max_workers=num_workers,
            )
        else:
            # マッピングベースで評価
            return evaluate_chord_with_mapping(
                predicted_chord_dir=predicted_chord_dir,
                reference_chord_dir=reference_chord_dir,
                track_mapping=track_mapping,
                frame_rate=chord_frame_rate,
                ignore_label=chord_ignore_label,
            )

    # ThreadPoolExecutorで並列実行
    with ThreadPoolExecutor(max_workers=3) as executor:
        fad_future = executor.submit(run_fad_eval)
        clap_future = executor.submit(run_clap_eval)
        chord_future = executor.submit(run_chord_eval)

        fad_score = fad_future.result()
        clap_result = clap_future.result()
        chord_result = chord_future.result()

    # 結果を構築
    overall_metrics: dict = {
        "fad_score": fad_score,
        "clap_score": clap_result.get("overall_clap_score", 0.0),
    }

    # 和音評価結果を追加
    if chord_result is not None:
        overall_metrics.update(chord_result.overall)

    # ファイルごとの結果
    files_result: dict = {}

    # CLAP個別スコア
    for name, score in clap_result.get("files", {}).items():
        if name not in files_result:
            files_result[name] = {}
        files_result[name]["clap_score"] = score

    # 和音評価個別スコア
    if chord_result is not None:
        for file_name, file_metrics in chord_result.files.items():
            audio_name = file_name.replace(".lab", ".wav")
            if audio_name not in files_result:
                files_result[audio_name] = {}
            files_result[audio_name]["chord_metrics"] = {
                "metrics": file_metrics.metrics,
                "meta": asdict(file_metrics.meta),
                "breakdown": asdict(file_metrics.breakdown),
            }
            # CLAP個別スコアを和音結果にも追加
            clap_files = clap_result.get("files", {})
            if audio_name in clap_files:
                file_metrics.metrics["clap_score"] = clap_files[audio_name]

    result = {
        "model_type": "pretrained_baseline",
        "model_name": "stabilityai/stable-audio-open-1.0",
        "condition_type": "text_only",
        "chord_evaluation_enabled": do_chord_eval,
        "overall_metrics": overall_metrics,
        "files": files_result,
        "generated_audio_dir": str(generated_audio_dir),
        "reference_audio_dir": str(reference_audio_dir),
    }

    if do_chord_eval and chord_result is not None:
        result["missing_predictions"] = chord_result.missing_predictions

    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrained model evaluation (FAD, CLAP, and optional chord evaluation)",
    )
    parser.add_argument(
        "generated_audio_dir",
        type=Path,
        help="Directory containing generated audio files",
    )
    parser.add_argument(
        "reference_audio_dir",
        type=Path,
        help="Directory containing reference audio files",
    )
    parser.add_argument(
        "--predicted-chord-dir",
        type=Path,
        default=None,
        help="Directory containing predicted chord lab files (optional, for chord evaluation)",
    )
    parser.add_argument(
        "--reference-chord-dir",
        type=Path,
        default=None,
        help="Directory containing reference chord lab files (optional, for chord evaluation)",
    )
    parser.add_argument(
        "--chord-frame-rate",
        type=float,
        default=24.0,
        help="Frame rate used for chord evaluation in Hz (default: 24.0)",
    )
    parser.add_argument(
        "--chord-ignore-label",
        type=str,
        default=None,
        help="Chord label to ignore during evaluation",
    )
    parser.add_argument(
        "--fad-model",
        type=str,
        default="vggish",
        help="Model name used for FAD calculation (default: vggish)",
    )
    parser.add_argument(
        "--clap-model-path",
        type=str,
        default="ckpts/music_audioset_epoch_15_esc_90.14.pt",
        help="Path to CLAP model weights",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for audio loading (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/pretrained_evaluation_results.json"),
        help="Path to write JSON results",
    )
    return parser.parse_args(argv)


def ensure_directory(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    ensure_directory(args.generated_audio_dir, "generated audio directory")
    ensure_directory(args.reference_audio_dir, "reference audio directory")

    result = evaluate_pretrained(
        generated_audio_dir=args.generated_audio_dir,
        reference_audio_dir=args.reference_audio_dir,
        predicted_chord_dir=args.predicted_chord_dir,
        reference_chord_dir=args.reference_chord_dir,
        chord_frame_rate=args.chord_frame_rate,
        chord_ignore_label=args.chord_ignore_label,
        fad_model_name=args.fad_model,
        clap_model_path=args.clap_model_path,
        num_workers=args.num_workers,
    )

    # 結果を表示
    print("\n" + "=" * 50)
    print("評価結果サマリー")
    print("=" * 50)
    print(f"モデル: {result['model_name']}")
    print(f"条件付け: {result['condition_type']}")
    print(f"和音評価: {'有効' if result['chord_evaluation_enabled'] else '無効'}")
    print("-" * 50)
    print(f"FAD Score: {result['overall_metrics']['fad_score']:.4f}")
    print(f"CLAP Score: {result['overall_metrics']['clap_score']:.4f}")

    if result["chord_evaluation_enabled"]:
        om = result["overall_metrics"]
        if "frame_accuracy" in om:
            print(f"Frame Accuracy: {om['frame_accuracy']:.4f}")
        if "root_accuracy" in om:
            print(f"Root Accuracy: {om['root_accuracy']:.4f}")
        if "quality_accuracy" in om:
            print(f"Quality Accuracy: {om['quality_accuracy']:.4f}")

    print(f"評価ファイル数: {len(result['files'])}")
    print("=" * 50)

    # JSONに保存
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")

    print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
