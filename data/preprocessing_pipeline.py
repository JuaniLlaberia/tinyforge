import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq

from config import DataConfig
from data.cleanup.mask_pii import mask_emails, mask_ips, mask_phone_numbers
from data.deduplication.exact_deduplication import exact_dedup_streaming
from data.deduplication.minhash_deduplication import minhash_dedup_streaming
from data.filters.gopher_quality_filter import gopher_quality_filter
from data.progress import format_duration
from data.split_dataset import write_parquet

def filter_doc(text: str) -> tuple[str | None, str | None]:
    """
    Run a document through the active filter pipeline: Gopher-style quality
    heuristics, then PII masking. Language-ID and the learned quality
    classifier are intentionally excluded here -- FineWeb-Edu already
    applied both upstream; see `data/measure_pipeline.py` to audit them
    as a diagnostic instead.

    Args:
        text (str): Document text.

    Returns:
        tuple[str | None, str | None]: (masked_text, None) if kept, or
        (None, drop_reason) if dropped.
    """
    if not gopher_quality_filter(text=text):
        return None, "gopher"

    masked_text, _ = mask_emails(text=text)
    masked_text, _ = mask_phone_numbers(text=masked_text)
    masked_text, _ = mask_ips(text=masked_text)

    return masked_text, None

def filter_split(input_dir: Path, output_dir: Path, limit: int | None) -> tuple[int, dict[str, int]]:
    """
    Filter every chunk in a split directory, streaming survivors straight to
    `output_dir` one chunk at a time -- never accumulates the whole split in
    memory (holding ~2.9M full documents this way was measured at ~28 GB).

    Args:
        input_dir (Path): Directory of chunk_*.parquet files.
        output_dir (Path): Directory to write surviving chunk_*.parquet files to (same filenames).
        limit (int | None): Max number of input chunk files to process.
    Returns:
        tuple[int, dict[str, int]]: Total kept count, and a count of dropped
        documents per drop reason.
    """
    chunk_files = sorted(input_dir.glob("chunk_*.parquet"))
    if limit is not None:
        chunk_files = chunk_files[:limit]

    drop_counts: dict[str, int] = {}
    kept_total = 0
    start_time = time.monotonic()

    for file_idx, chunk_file in enumerate(chunk_files):
        survivors: list[dict] = []
        for row in pq.read_table(chunk_file).to_pylist():
            masked_text, reason = filter_doc(row["text"])
            if reason is not None:
                drop_counts[reason] = drop_counts.get(reason, 0) + 1
                continue
            row["text"] = masked_text
            survivors.append(row)

        if survivors:
            batch = {col: [doc[col] for doc in survivors] for col in DataConfig.keep_columns}
            write_parquet(batch, output_dir / chunk_file.name)
        kept_total += len(survivors)

        elapsed = time.monotonic() - start_time
        rate = (file_idx + 1) / elapsed if elapsed > 0 else 0.0
        eta = (len(chunk_files) - (file_idx + 1)) / rate if rate > 0 else None
        print(
            f"Filter {input_dir.name} | {chunk_file.name} | {file_idx + 1}/{len(chunk_files)} files | "
            f"{kept_total} kept so far | elapsed {format_duration(elapsed)} | "
            f"eta {format_duration(eta) if eta is not None else 'unknown'}"
        )

    return kept_total, drop_counts

def _stage_complete(input_dir: Path, output_dir: Path) -> bool:
    """
    Check whether a streaming stage already finished, so `run()` can skip it
    on a resumed invocation instead of redoing already-correct work.

    Args:
        input_dir (Path): Directory the stage reads chunk_*.parquet files from.
        output_dir (Path): Directory the stage writes chunk_*.parquet files to.
    Returns:
        bool: True if every input chunk has a same-named output chunk.
    """
    input_files = sorted(input_dir.glob("chunk_*.parquet"))
    if not input_files:
        return False
    return all((output_dir / f.name).exists() for f in input_files)

def _count_docs(directory: Path) -> int:
    """
    Count total documents across chunk_*.parquet files in a directory.

    Args:
        directory (Path): Directory of chunk_*.parquet files.
    Returns:
        int: Total row count.
    """
    return sum(pq.ParquetFile(f).metadata.num_rows for f in directory.glob("chunk_*.parquet"))

def run(limit: int | None = None) -> None:
    """
    Run Step 3 (filter) then Step 4 (dedup: exact then MinHash) over train +
    val. Every stage streams chunk-by-chunk and persists its own
    intermediate output, so memory stays bounded regardless of corpus size,
    and a resumed invocation skips any stage that already finished rather
    than redoing it (MinHash itself still reruns in full if resumed mid-way
    through it -- only filter and exact dedup checkpoint at the file level).

    Args:
        limit (int | None): Max input chunk files per split, for testing.
    Returns:
        None.
    """
    filtered_train_dir = DataConfig.filtered_path / "train"
    filtered_val_dir = DataConfig.filtered_path / "val"
    if _stage_complete(DataConfig.train_path, filtered_train_dir) and _stage_complete(DataConfig.val_path, filtered_val_dir):
        train_kept = _count_docs(filtered_train_dir)
        val_kept = _count_docs(filtered_val_dir)
        print(f"Filter | already done, skipping | train {train_kept} | val {val_kept}")
    else:
        train_kept, train_drops = filter_split(DataConfig.train_path, filtered_train_dir, limit)
        val_kept, val_drops = filter_split(DataConfig.val_path, filtered_val_dir, limit)
        print(f"Filter | train {train_kept} kept | dropped {train_drops}")
        print(f"Filter | val {val_kept} kept | dropped {val_drops}")

    exact_train_dir = DataConfig.exact_dedup_path / "train"
    exact_val_dir = DataConfig.exact_dedup_path / "val"
    if _stage_complete(filtered_train_dir, exact_train_dir) and _stage_complete(filtered_val_dir, exact_val_dir):
        exact_train_count = _count_docs(exact_train_dir)
        exact_val_count = _count_docs(exact_val_dir)
        print(f"Exact dedup | already done, skipping | train {exact_train_count} | val {exact_val_count}")
    else:
        exact_train_count, exact_val_count = exact_dedup_streaming(
            val_input_dir=filtered_val_dir,
            train_input_dir=filtered_train_dir,
            val_output_dir=exact_val_dir,
            train_output_dir=exact_train_dir,
        )
        print(f"Exact dedup | train {exact_train_count} | val {exact_val_count}")

    final_train_dir = DataConfig.preprocessing_output_path / "train"
    final_val_dir = DataConfig.preprocessing_output_path / "val"
    final_train_count, final_val_count = minhash_dedup_streaming(
        val_input_dir=exact_val_dir,
        train_input_dir=exact_train_dir,
        val_output_dir=final_val_dir,
        train_output_dir=final_train_dir,
        num_hashes=DataConfig.minhash_num_hashes,
        num_bands=DataConfig.minhash_num_bands,
        ngram_length=DataConfig.minhash_ngram_length,
        docs_per_chunk=DataConfig.docs_per_chunk,
    )
    print(f"MinHash dedup | train {final_train_count} | val {final_val_count}")

    print(f"DONE | preprocessing complete | train {final_train_count} | val {final_val_count}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter (Gopher + PII) then dedup (exact + MinHash) train/val.")
    parser.add_argument("--limit", type=int, default=None, help="Max input chunk files per split (testing only).")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    run(limit=args.limit)

if __name__ == "__main__":
    main()
