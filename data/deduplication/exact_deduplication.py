import time
from pathlib import Path

import pyarrow.parquet as pq
import xxhash

from config import DataConfig
from data.progress import format_duration
from data.split_dataset import write_parquet

def generate_hash(text: str) -> int:
    """
    Compute a 64-bit hash of a line of text.

    Args:
        text (str): Input text.

    Returns:
        int: 64-bit integer hash of the text.
    """
    return xxhash.xxh64(text.encode("utf-8")).intdigest()

def exact_line_deduplication(paths: list[Path], output_path: Path) -> Path:
    """
    Remove lines that appear more than once across a set of files.

    Args:
        paths (list[Path]): Input files to deduplicate, treated as one combined corpus.
        output_path (Path): Directory to write deduplicated copies of each file to.

    Returns:
        Path: The output directory containing the deduplicated files.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    unique_count = {}
    for path in paths:
        with open(path, "r") as file:
            for line in file:
                hash = generate_hash(line)
                unique_count[hash] = unique_count.get(hash, 0) + 1

    for path in paths:
        new_path = output_path / path.name
        with open(new_path, "w") as out:
            with open(path, "r") as file:
                for line in file:
                    if unique_count[generate_hash(line)] == 1:
                        out.write(line)

    return output_path

def normalize_for_hash(text: str) -> str:
    """
    Collapse whitespace and lowercase text for exact-duplicate hashing.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text.
    """
    return " ".join(text.split()).lower()

def exact_document_deduplication(train_docs: list[dict], val_docs: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Remove exact-duplicate documents (by hash of normalized text) across
    train and val combined. Val is processed first, so when a document
    appears in both, the train copy is the one dropped -- this is what
    prevents train/val leakage from exact duplicates.

    Args:
        train_docs (list[dict]): Train documents, each with a "text" field.
        val_docs (list[dict]): Val documents, each with a "text" field.

    Returns:
        tuple[list[dict], list[dict]]: (train_survivors, val_survivors).
    """
    seen_hashes: set[int] = set()

    val_survivors = []
    for doc in val_docs:
        doc_hash = generate_hash(normalize_for_hash(doc["text"]))
        if doc_hash in seen_hashes:
            continue
        seen_hashes.add(doc_hash)
        val_survivors.append(doc)

    train_survivors = []
    for doc in train_docs:
        doc_hash = generate_hash(normalize_for_hash(doc["text"]))
        if doc_hash in seen_hashes:
            continue
        seen_hashes.add(doc_hash)
        train_survivors.append(doc)

    return train_survivors, val_survivors

def _dedup_stream(input_dir: Path, output_dir: Path, seen_hashes: set[int]) -> int:
    """
    Stream chunk_*.parquet files, dropping rows whose normalized-text hash
    is already in `seen_hashes` (updated in place as new hashes are seen),
    writing survivors to `output_dir` with the same filenames. Never holds
    more than one chunk's documents in memory.

    Args:
        input_dir (Path): Directory of chunk_*.parquet files to dedup.
        output_dir (Path): Directory to write deduped chunks to.
        seen_hashes (set[int]): Hashes already seen; shared and updated across calls.

    Returns:
        int: Total surviving documents.
    """
    chunk_files = sorted(input_dir.glob("chunk_*.parquet"))
    total = 0
    start_time = time.monotonic()

    for file_idx, chunk_file in enumerate(chunk_files):
        rows = pq.read_table(chunk_file).to_pylist()
        survivors = []
        for row in rows:
            doc_hash = generate_hash(normalize_for_hash(row["text"]))
            if doc_hash in seen_hashes:
                continue
            seen_hashes.add(doc_hash)
            survivors.append(row)

        if survivors:
            batch = {col: [doc[col] for doc in survivors] for col in DataConfig.keep_columns}
            write_parquet(batch, output_dir / chunk_file.name)
        total += len(survivors)

        elapsed = time.monotonic() - start_time
        print(
            f"Exact dedup {input_dir.name} | {chunk_file.name} | {file_idx + 1}/{len(chunk_files)} files | "
            f"{total} kept so far | elapsed {format_duration(elapsed)}"
        )

    return total

def exact_dedup_streaming(
    val_input_dir: Path,
    train_input_dir: Path,
    val_output_dir: Path,
    train_output_dir: Path,
) -> tuple[int, int]:
    """
    Remove exact-duplicate documents (by hash of normalized text) across
    train and val combined, streaming chunk by chunk from disk to disk. Val
    is processed first, so when a document appears in both, the train copy
    is the one dropped -- this is what prevents train/val leakage from
    exact duplicates. Never holds more than one chunk's documents in
    memory; only `seen_hashes` (small 64-bit ints) persists across chunks.

    Args:
        val_input_dir (Path): Directory of filtered val chunk_*.parquet files.
        train_input_dir (Path): Directory of filtered train chunk_*.parquet files.
        val_output_dir (Path): Directory to write deduped val chunks to.
        train_output_dir (Path): Directory to write deduped train chunks to.

    Returns:
        tuple[int, int]: (train survivor count, val survivor count).
    """
    seen_hashes: set[int] = set()

    val_count = _dedup_stream(val_input_dir, val_output_dir, seen_hashes)
    train_count = _dedup_stream(train_input_dir, train_output_dir, seen_hashes)

    return train_count, val_count