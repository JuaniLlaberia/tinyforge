import random
from pathlib import Path

import pyarrow.parquet as pq

from config import DataConfig, GlobalConfig, TokenizerConfig

def prepare_sample(input_dir: Path, output_path: Path, target_bytes: int, seed: int) -> int:
    """
    Sample documents from filtered/deduped train chunks and write them to a
    single <|endoftext|>-delimited text file for BPE training.

    Args:
        input_dir (Path): Directory of chunk_*.parquet files to sample from.
        output_path (Path): Where to write the sample text file.
        target_bytes (int): Approximate sample size to stop at.
        seed (int): Random seed, so the sample is reproducible.
    Returns:
        int: Actual size of the written file, in bytes.
    """
    chunk_files = sorted(input_dir.glob("chunk_*.parquet"))
    rng = random.Random(seed)
    rng.shuffle(chunk_files)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_bytes = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for chunk_file in chunk_files:
            if written_bytes >= target_bytes:
                break

            texts = pq.read_table(chunk_file, columns=["text"]).column("text").to_pylist()
            rng.shuffle(texts)

            for text in texts:
                if written_bytes >= target_bytes:
                    break
                piece = text + "<|endoftext|>"
                out.write(piece)
                written_bytes += len(piece.encode("utf-8"))

    return written_bytes

def main() -> None:
    written = prepare_sample(
        input_dir=DataConfig.preprocessing_output_path / "train",
        output_path=TokenizerConfig.sample_path,
        target_bytes=TokenizerConfig.sample_bytes,
        seed=GlobalConfig.seed,
    )
    print(f"Wrote {written / 1e6:.1f} MB to {TokenizerConfig.sample_path}")

if __name__ == "__main__":
    main()
