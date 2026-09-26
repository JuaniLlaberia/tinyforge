import argparse
import random
import time
from pathlib import Path

import pyarrow.parquet as pq

from config import DataConfig, GlobalConfig, TokenizerConfig
from data.progress import format_duration
from tokenizer.tokenizer import Tokenizer
from tokenizer.tokenizer_hf import build_hf_tokenizer

def sample_docs(input_dir: Path, n: int, seed: int) -> list[str]:
    """
    Randomly sample document texts across every chunk in a directory.

    Args:
        input_dir (Path): Directory of chunk_*.parquet files.
        n (int): Number of documents to sample.
        seed (int): Random seed, so the sample is reproducible.
    Returns:
        list[str]: Sampled document texts.
    """
    chunk_files = sorted(input_dir.glob("chunk_*.parquet"))
    rng = random.Random(seed)
    rng.shuffle(chunk_files)

    docs: list[str] = []
    for chunk_file in chunk_files:
        if len(docs) >= n:
            break
        texts = pq.read_table(chunk_file, columns=["text"]).column("text").to_pylist()
        rng.shuffle(texts)
        docs.extend(texts[: n - len(docs)])

    return docs

def verify(n: int, seed: int) -> bool:
    """
    Encode a sample of real train docs with both the A1 encoder and the
    fast HF encoder, and confirm the token id sequences are identical.

    Args:
        n (int): Number of documents to check.
        seed (int): Random seed for sampling.
    Returns:
        bool: True if every sampled doc produced identical ids on both encoders.
    """
    docs = sample_docs(DataConfig.preprocessing_output_path / "train", n, seed)
    print(f"Verifying {len(docs)} docs")

    a1 = Tokenizer.from_files(str(TokenizerConfig.vocab_path), str(TokenizerConfig.merges_path), list(TokenizerConfig.special_tokens))
    hf = build_hf_tokenizer()

    mismatches = 0
    start_time = time.monotonic()
    for i, text in enumerate(docs):
        a1_ids = a1.encode(text)
        hf_ids = hf.encode(text).ids

        if a1_ids != hf_ids:
            mismatches += 1
            if mismatches <= 5:
                first_diff = next((j for j in range(min(len(a1_ids), len(hf_ids))) if a1_ids[j] != hf_ids[j]), min(len(a1_ids), len(hf_ids)))
                print(f"MISMATCH doc {i} | a1 {len(a1_ids)} ids | hf {len(hf_ids)} ids | diverge at index {first_diff}")
                print(f"  a1: {a1_ids[max(0, first_diff - 3):first_diff + 5]}")
                print(f"  hf: {hf_ids[max(0, first_diff - 3):first_diff + 5]}")

        if (i + 1) % 1000 == 0 or i + 1 == len(docs):
            elapsed = time.monotonic() - start_time
            print(f"Verify | {i + 1}/{len(docs)} docs | {mismatches} mismatches | elapsed {format_duration(elapsed)}")

    if mismatches == 0:
        print(f"PASS | all {len(docs)} docs produced identical ids")
    else:
        print(f"FAIL | {mismatches}/{len(docs)} docs mismatched")

    return mismatches == 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the fast HF encoder matches the A1 encoder exactly.")
    parser.add_argument("--n", type=int, default=10_000, help="Number of docs to check.")
    parser.add_argument("--seed", type=int, default=GlobalConfig.seed)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    ok = verify(n=args.n, seed=args.seed)
    if not ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
