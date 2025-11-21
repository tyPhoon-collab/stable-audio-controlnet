import functools

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
    if model.enable_fusion:
        # (1, Time)
        audio_input_tensor = audio_data.reshape(1, -1)
    elif use_segment_averaging:
        CHUNK_SAMPLES = SR * 10  # 10秒
        total_samples = len(audio_data)
        chunks = []

        # バッチ作成ループ
        for start in range(0, total_samples, CHUNK_SAMPLES):
            end = start + CHUNK_SAMPLES
            chunk = audio_data[start:end]
            # パディングして長さを揃える
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)), "constant")
            chunks.append(chunk)

        if not chunks:
            return 0.0

        audio_input_tensor = np.array(chunks)  # (Batch, 480000)
    else:
        audio_input_tensor = audio_data.reshape(1, -1)

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
