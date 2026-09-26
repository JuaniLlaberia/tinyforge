import argparse
import hashlib
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from config import DataConfig

def is_val_doc(doc_id: str, val_mod: int) -> bool:
    """
    Deterministically decide whether a document belongs to the validation split.

    Args:
        doc_id (str): Document id.
        val_mod (int): Modulus; roughly 1/val_mod of documents go to val.
    Returns:
        bool: True if the document belongs in the validation split.
    """
    digest = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % val_mod == 0

def split_batch(batch: dict, val_mod: int) -> tuple[dict, dict]:
    """
    Split a column-oriented batch of documents into train and val batches.

    Args:
        batch (dict): Column-oriented batch, e.g. {"id": [...], "text": [...], ...}.
        val_mod (int): Modulus passed to `is_val_doc`.
    Returns:
        tuple[dict, dict]: (train_batch, val_batch), same columns as `batch`.
    """
    is_val = [is_val_doc(doc_id, val_mod) for doc_id in batch["id"]]

    train_batch = {col: [v for v, val in zip(values, is_val) if not val] for col, values in batch.items()}
    val_batch = {col: [v for v, val in zip(values, is_val) if val] for col, values in batch.items()}

    return train_batch, val_batch

def write_parquet(batch: dict, path: Path) -> None:
    """
    Write a column-oriented batch to a parquet file atomically.

    Args:
        batch (dict): Column-oriented batch to write.
        path (Path): Destination file path.
    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.table(batch), tmp, compression="zstd")
    os.replace(tmp, path)

def split_dataset(raw_path: Path, train_path: Path, val_path: Path, val_mod: int, limit: int | None) -> None:
    """
    Split every downloaded chunk into train/val by a deterministic hash of doc id.

    Args:
        raw_path (Path): Directory containing downloaded `chunk_*.parquet` files.
        train_path (Path): Directory to write train chunks to.
        val_path (Path): Directory to write the combined val chunk to.
        val_mod (int): Modulus passed to `is_val_doc`.
        limit (int | None): Max number of input chunk files to process, for testing.
    Returns:
        None.
    """
    chunk_files = sorted(raw_path.glob("chunk_*.parquet"))
    if limit is not None:
        chunk_files = chunk_files[:limit]

    val_accumulator = {col: [] for col in DataConfig.keep_columns}
    train_docs = 0
    val_docs = 0

    for chunk_file in chunk_files:
        batch = pq.read_table(chunk_file).to_pydict()
        train_batch, val_batch = split_batch(batch, val_mod)

        write_parquet(train_batch, train_path / chunk_file.name)
        for col in DataConfig.keep_columns:
            val_accumulator[col].extend(val_batch[col])

        train_docs += len(train_batch["id"])
        val_docs += len(val_batch["id"])
        print(f"{chunk_file.name} | train {train_docs} | val {val_docs}")

    if val_accumulator["id"]:
        write_parquet(val_accumulator, val_path / "chunk_00000.parquet")

    print(f"Done | {train_docs} train docs | {val_docs} val docs")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split downloaded chunks into train/val by hashed doc id.")
    parser.add_argument("--input", type=Path, default=DataConfig.output_path, help="Directory with raw chunk_*.parquet files.")
    parser.add_argument("--train-output", type=Path, default=DataConfig.train_path, help="Directory to write train chunks to.")
    parser.add_argument("--val-output", type=Path, default=DataConfig.val_path, help="Directory to write the combined val chunk to.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of input chunk files to process (testing only).")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    split_dataset(
        raw_path=args.input,
        train_path=args.train_output,
        val_path=args.val_output,
        val_mod=DataConfig.val_split_mod,
        limit=args.limit,
    )

if __name__ == "__main__":
    main()
