# soundfileのインストールがうまくいかない場合がある。
# このとき、torchaudio.loadで音声ファイルを読み込むとエラーが出る。
# ちゃんとロードできるか確認するためのスクリプト

import torchaudio


def torchaudio_load_test():
    test_data_path = "data/A Classic Education - NightOwl.other.mp3"

    data, sr = torchaudio.load(test_data_path)
    print(f"Loaded audio data with shape: {data.shape}, Sample rate: {sr}")


if __name__ == "__main__":
    torchaudio_load_test()
