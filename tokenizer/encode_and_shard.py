import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from config import DataConfig, TokenizerConfig
from data.progress import format_duration
from tokenizer.tokenizer_hf import build_hf_tokenizer

class ShardWriter:
    def __init__(self, output_dir: Path, shard_tokens: int):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.buffer = np.empty(0, dtype=np.uint32)
        self.shard_id = 0
        self.total_tokens = 0

    def add(self, ids: list[int]) -> None:
        """
        Add a batch of token ids to the buffer, flushing full shards as needed.

        Args:
            ids (list[int]): Token ids to append.
        Returns:
            None.
        """
        self.buffer = np.concatenate([self.buffer, np.asarray(ids, dtype=np.uint32)])
        while len(self.buffer) >= self.shard_tokens:
            self._flush(self.buffer[: self.shard_tokens])
            self.buffer = self.buffer[self.shard_tokens :]

    def _flush(self, ids: np.ndarray) -> None:
        """
        Write a shard of token ids to disk as raw uint16.

        Args:
            ids (np.ndarray): Token ids to write, length <= shard_tokens.
        Returns:
            None.
        """
        assert len(ids) == 0 or int(ids.max()) < 65536, f"token id {int(ids.max())} exceeds uint16 range"

        path = self.output_dir / f"shard_{self.shard_id:04d}.bin"
        ids.astype(np.uint16).tofile(path)
        self.total_tokens += len(ids)
        self.shard_id += 1

    def close(self) -> None:
        """
        Flush any remaining buffered tokens as a final, possibly short, shard.

        Returns:
            None.
        """
        if len(self.buffer):
            self._flush(self.buffer)
            self.buffer = np.empty(0, dtype=np.uint32)

def encode_split(input_dir: Path, output_dir: Path, tokenizer: Tokenizer, eot_id: int, shard_tokens: int, limit: int | None) -> int:
    """
    Encode every chunk in a split directory and write uint16 shards.

    Args:
        input_dir (Path): Directory of chunk_*.parquet files to encode.
        output_dir (Path): Directory to write shard_*.bin files to.
        tokenizer (Tokenizer): Fast HF encoder, already verified against A1.
        eot_id (int): Token id appended after every document.
        shard_tokens (int): Tokens per output shard.
        limit (int | None): Max number of input chunk files to process.
    Returns:
        int: Total tokens written.
    """
    chunk_files = sorted(input_dir.glob("chunk_*.parquet"))
    if limit is not None:
        chunk_files = chunk_files[:limit]

    writer = ShardWriter(output_dir, shard_tokens)
    start_time = time.monotonic()
    docs_done = 0

    for file_idx, chunk_file in enumerate(chunk_files):
        texts = pq.read_table(chunk_file, columns=["text"]).column("text").to_pylist()
        encodings = tokenizer.encode_batch(texts)

        chunk_ids: list[int] = []
        for encoding in encodings:
            chunk_ids.extend(encoding.ids)
            chunk_ids.append(eot_id)
        writer.add(chunk_ids)

        docs_done += len(texts)
        elapsed = time.monotonic() - start_time
        rate = docs_done / elapsed if elapsed > 0 else 0.0
        eta = (len(chunk_files) - (file_idx + 1)) * (elapsed / (file_idx + 1)) if file_idx + 1 > 0 else None
        print(
            f"Encode {input_dir.name} | {chunk_file.name} | {file_idx + 1}/{len(chunk_files)} files | "
            f"{docs_done} docs | {writer.total_tokens + len(writer.buffer)} tokens | "
            f"{rate:.0f} docs/s | elapsed {format_duration(elapsed)} | "
            f"eta {format_duration(eta) if eta is not None else 'unknown'}"
        )

    writer.close()
    print(f"Encode {input_dir.name} | done | {writer.total_tokens} tokens in {writer.shard_id} shards")
    return writer.total_tokens

def run(limit: int | None = None) -> None:
    """
    Encode train + val and write uint16 shards.

    Args:
        limit (int | None): Max input chunk files per split, for testing.
    Returns:
        None.
    """
    tokenizer = build_hf_tokenizer()
    eot_id = TokenizerConfig.special_tokens.index("<|endoftext|>") + 256
    assert eot_id == TokenizerConfig.eot_id, f"EOT id mismatch: vocab has it at {eot_id}, config says {TokenizerConfig.eot_id}"

    train_tokens = encode_split(
        input_dir=DataConfig.preprocessing_output_path / "train",
        output_dir=TokenizerConfig.shards_path / "train",
        tokenizer=tokenizer,
        eot_id=eot_id,
        shard_tokens=TokenizerConfig.shard_tokens,
        limit=limit,
    )
    val_tokens = encode_split(
        input_dir=DataConfig.preprocessing_output_path / "val",
        output_dir=TokenizerConfig.shards_path / "val",
        tokenizer=tokenizer,
        eot_id=eot_id,
        shard_tokens=TokenizerConfig.shard_tokens,
        limit=limit,
    )

    print(f"DONE | encode + shard complete | train {train_tokens} tokens | val {val_tokens} tokens")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode filtered/deduped train+val and write uint16 token shards.")
    parser.add_argument("--limit", type=int, default=None, help="Max input chunk files per split (testing only).")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    run(limit=args.limit)

if __name__ == "__main__":
    main()
