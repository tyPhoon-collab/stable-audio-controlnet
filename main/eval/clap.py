import functools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import laion_clap
import librosa
import numpy as np
import torch


def initialize_clap_model(
    weights_path: str = "ckpts/music_audioset_epoch_15_esc_90.14.pt",
    amodel: str = "HTSAT-base",
    device: str = "cuda",
    enable_fusion: bool = False,  # 引数で指定できるように変更
) -> laion_clap.CLAP_Module:
    """
    CLAPモデルを初期化します。
    汎用モデルを使う場合は enable_fusion=True を指定してください。
    音楽特化モデル(music_audioset...)の場合は False にする必要があります。
    """
    model = laion_clap.CLAP_Module(
        enable_fusion=enable_fusion,
        device=device,
        amodel=amodel,
    )

    # torch.load の安全対策パッチ
    _original_torch_load = torch.load

    @functools.wraps(_original_torch_load)
    def unsafe_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    try:
        torch.load = unsafe_torch_load
        model.load_ckpt(weights_path)
    finally:
        torch.load = _original_torch_load

    model.to(device)
    model.eval()

    return model


def _load_audio(audio_path: str | Path, sr: int = 48000) -> np.ndarray | None:
    """音声ファイルを読み込む（並列化用）"""
    try:
        audio_data, _ = librosa.load(str(audio_path), sr=sr)
        return audio_data
    except Exception:
        return None


def _prepare_audio_tensor(
    audio_data: np.ndarray,
    enable_fusion: bool,
    use_segment_averaging: bool,
    sr: int = 48000,
) -> np.ndarray:
    """音声データをモデル入力用のテンソルに変換"""
    if enable_fusion:
        return audio_data.reshape(1, -1)

    if use_segment_averaging:
        chunk_samples = sr * 10  # 10秒
        total_samples = len(audio_data)
        chunks = []

        for start in range(0, total_samples, chunk_samples):
            end = start + chunk_samples
            chunk = audio_data[start:end]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), "constant")
            chunks.append(chunk)

        if not chunks:
            return np.zeros((1, chunk_samples))

        return np.array(chunks)

    return audio_data.reshape(1, -1)


def calculate_pair_similarity(
    model: laion_clap.CLAP_Module,
    audio_input: str | np.ndarray,
    text: str,
    device: str = "cuda",
    use_segment_averaging: bool = True,  # チャンク化して平均をとるかどうかのフラグ
) -> float:
    """
    単一のオーディオと単一のテキストのコサイン類似度を計算します。

    Args:
        use_segment_averaging (bool):
            TrueかつモデルがFusion非対応の場合、長い音声を10秒ごとに分割して平均をとります。
            Falseの場合、冒頭10秒のみを使用します。
            (モデル自体がFusion対応の場合は、このフラグに関わらずモデルのFusion機能が優先されます)
    """
    # 1. オーディオのロード
    SR = 48000
    audio_data: np.ndarray
    if isinstance(audio_input, str):
        audio_data, _ = librosa.load(audio_input, sr=SR)
    else:
        audio_data = audio_input

    # 2. オーディオ入力テンソルの準備
    audio_input_tensor = _prepare_audio_tensor(
        audio_data, model.enable_fusion, use_segment_averaging, SR
    )

    # 3. 埋め込みの取得（全ケースで共通）
    with torch.no_grad():
        text_embed_np = model.get_text_embedding([text])
        audio_embed_np = model.get_audio_embedding_from_data(
            x=audio_input_tensor, use_tensor=False
        )

        text_tensor = torch.from_numpy(text_embed_np).to(device)
        audio_tensor = torch.from_numpy(audio_embed_np).to(device)

        # 平均化 (Mean Pooling) - Manual Chunking の場合のみ適用
        if (
            use_segment_averaging
            and not model.enable_fusion
            and audio_tensor.shape[0] > 1
        ):
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

    # 4. 正規化と類似度計算
    text_tensor = text_tensor / text_tensor.norm(dim=-1, keepdim=True)
    audio_tensor = audio_tensor / audio_tensor.norm(dim=-1, keepdim=True)
    similarity = (text_tensor @ audio_tensor.T).item()

    return similarity


def calculate_batch_similarity(
    model: laion_clap.CLAP_Module,
    audio_data_list: list[np.ndarray],
    texts: list[str],
    device: str = "cuda",
    use_segment_averaging: bool = True,
) -> list[float]:
    """
    複数の音声とテキストのペアでコサイン類似度をバッチ計算します。

    Args:
        model: CLAPモデル
        audio_data_list: 音声データのリスト（既に読み込み済み）
        texts: テキストのリスト
        device: デバイス
        use_segment_averaging: チャンク化して平均をとるか

    Returns:
        類似度のリスト
    """
    if len(audio_data_list) != len(texts):
        raise ValueError("audio_data_list と texts の長さが一致しません")

    if not audio_data_list:
        return []

    SR = 48000

    # テキストembeddingをバッチで取得
    with torch.no_grad():
        text_embed_np = model.get_text_embedding(texts)
        text_tensor = torch.from_numpy(text_embed_np).to(device)
        text_tensor = text_tensor / text_tensor.norm(dim=-1, keepdim=True)

    # 音声embeddingを個別に取得（チャンク処理があるため）
    similarities: list[float] = []

    for i, audio_data in enumerate(audio_data_list):
        audio_input_tensor = _prepare_audio_tensor(
            audio_data, model.enable_fusion, use_segment_averaging, SR
        )

        with torch.no_grad():
            audio_embed_np = model.get_audio_embedding_from_data(
                x=audio_input_tensor, use_tensor=False
            )
            audio_tensor = torch.from_numpy(audio_embed_np).to(device)

            # 平均化 (Mean Pooling)
            if (
                use_segment_averaging
                and not model.enable_fusion
                and audio_tensor.shape[0] > 1
            ):
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            audio_tensor = audio_tensor / audio_tensor.norm(dim=-1, keepdim=True)

            # 対応するテキストとの類似度を計算
            similarity = (text_tensor[i : i + 1] @ audio_tensor.T).item()
            similarities.append(similarity)

    return similarities


def evaluate_clap_batch(
    model: laion_clap.CLAP_Module,
    audio_paths: list[Path],
    prompts: list[str],
    device: str = "cuda",
    num_workers: int = 4,
    use_segment_averaging: bool = True,
) -> dict[str, float]:
    """
    複数の音声ファイルとプロンプトのCLAPスコアをバッチ計算します。

    Args:
        model: 事前ロード済みCLAPモデル
        audio_paths: 音声ファイルパスのリスト
        prompts: 対応するプロンプトのリスト
        device: デバイス
        num_workers: 音声読み込みの並列数
        use_segment_averaging: チャンク化して平均をとるか

    Returns:
        ファイル名 -> スコア の辞書
    """
    if len(audio_paths) != len(prompts):
        raise ValueError("audio_paths と prompts の長さが一致しません")

    if not audio_paths:
        return {}

    # 音声ファイルを並列で読み込み
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        audio_data_list = list(executor.map(_load_audio, audio_paths))

    # 有効なデータのみフィルタリング
    valid_indices: list[int] = []
    valid_audio_data: list[np.ndarray] = []
    valid_prompts: list[str] = []

    for i, audio_data in enumerate(audio_data_list):
        if audio_data is not None:
            valid_indices.append(i)
            valid_audio_data.append(audio_data)
            valid_prompts.append(prompts[i])

    # バッチで類似度計算
    similarities = calculate_batch_similarity(
        model=model,
        audio_data_list=valid_audio_data,
        texts=valid_prompts,
        device=device,
        use_segment_averaging=use_segment_averaging,
    )

    # 結果を辞書に格納
    result: dict[str, float] = {}
    for idx, sim in zip(valid_indices, similarities):
        result[audio_paths[idx].name] = sim

    return result


if __name__ == "__main__":
    # 1. 初期化
    model = initialize_clap_model()

    # 2. テストデータ
    description = "An upbeat track featuring acoustic guitars, rhythmic percussion, and bright piano, creating a lively and cheerful mood with a medium tempo."
    # description = "The dog is barking."
    audio_path = "data/Al James - Schoolboy Facination_no_vocals.mp3"

    # 3. 個別に実行
    score = calculate_pair_similarity(model, audio_path, description)

    # 4. 結果表示
    print("\n--- Single Pair Comparison Results ---")
    print(f"Score: {score:.4f}")
