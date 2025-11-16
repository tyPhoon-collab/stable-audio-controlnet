import os
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path

from frechet_audio_distance import FrechetAudioDistance
from frechet_audio_distance.utils import load_audio_task
from tqdm import tqdm


class FrechetAudioDistanceFiltered(FrechetAudioDistance):
    """音声ファイルのみをフィルタリングするカスタム実装"""

    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

    def _FrechetAudioDistance__load_audio_files(self, dir, dtype="float32"):
        audio_files = [
            f
            for f in os.listdir(dir)
            if Path(f).suffix.lower() in self.AUDIO_EXTENSIONS
        ]

        if len(audio_files) == 0:
            if self.verbose:
                print("[Frechet Audio Distance] No audio files found in {}".format(dir))
            return []

        task_results = []
        pool = ThreadPool(self.audio_load_worker)
        pbar = tqdm(total=len(audio_files), disable=(not self.verbose))

        def update(*a):
            pbar.update()

        if self.verbose:
            print("[Frechet Audio Distance] Loading audio from {}...".format(dir))

        for fname in audio_files:
            res = pool.apply_async(
                load_audio_task,
                args=(os.path.join(dir, fname), self.sample_rate, self.channels, dtype),
                callback=update,
            )
            task_results.append(res)

        pool.close()
        pool.join()

        return [k.get() for k in task_results]


def compute_fad_score(
    reference_dir: str,
    generated_dir: str,
    model_name: str = "vggish",
    use_pca: bool = False,
    use_activation: bool = False,
    verbose: bool = True,
) -> float:
    """
    Compute Frechet Audio Distance (FAD) between two audio directories.

    Args:
        reference_dir: Path to reference audio directory
        generated_dir: Path to generated audio directory
        model_name: Model name for FrechetAudioDistance (default: "vggish")
        use_pca: Whether to use PCA (default: False)
        use_activation: Whether to use activation (default: False)
        verbose: Whether to print verbose output (default: True)

    Returns:
        FAD score as float
    """
    frechet = FrechetAudioDistanceFiltered(
        model_name=model_name,
        use_pca=use_pca,
        use_activation=use_activation,
        verbose=verbose,
    )

    fad_score = frechet.score(reference_dir, generated_dir)
    return fad_score


if __name__ == "__main__":
    fad_score = compute_fad_score(
        reference_dir="/content/drive/MyDrive/research/data/temp/real",
        generated_dir="/content/drive/MyDrive/research/data/temp/gen",
    )
    print(f"FAD score: {fad_score}")
