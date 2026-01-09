import torchaudio
import glob

files = glob.glob("/home/shima/project/osawa/slakh2100/babyslakh_16k/Track00001/stems/*.wav")
for f in files[:3]:
    info = torchaudio.info(f)
    print(f"{f}: {info.num_channels}ch, {info.sample_rate}Hz")
