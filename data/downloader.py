import argparse
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from config import DataConfig
from data.progress import format_duration as _format_duration

class Downloader:
    def __init__(
        self,
        output_path: Path = DataConfig.output_path,
        target_tokens: int = DataConfig.target_tokens,
        docs_per_chunk: int = DataConfig.docs_per_chunk,
        limit: int | None = None,
    ):
        """
        Initializes the Downloader class.

        Args:
            output_path (Path): Dir where to store downloaded chunks and progress.json.
            target_tokens (int): Token budget to stop at.
            docs_per_chunk (int): Documents per streamed batch and per output chunk file.
            limit (int | None): Stop after this many documents instead of `target_tokens`.
        Returns:
            None.
        """
        self.output_path = Path(output_path)
        self.target_tokens = target_tokens
        self.docs_per_chunk = docs_per_chunk
        self.limit = limit

        self.output_path.mkdir(parents=True, exist_ok=True)

        self.progress_path = self.output_path / "progress.json"
        self.progress = self._load_progress()

        self.dataset = load_dataset(DataConfig.source, name=DataConfig.subset, split="train", streaming=True)
        if self.progress["state"] is not None:
            self.dataset.load_state_dict(self.progress["state"])

    def _load_progress(self) -> dict:
        """
        Loads progress dictionary.

        Returns:
            dict: Dictionary containing the progress information (chunks, docs, tokens, state).
        """
        if self.progress_path.exists():
            return json.loads(self.progress_path.read_text())

        return {"chunks": 0, "docs": 0, "tokens": 0, "state": None}

    def _save_progress(self) -> None:
        """
        Stores progress dictionary.

        Returns:
            None.
        """
        self.progress_path.write_text(json.dumps(self.progress))

    def _write_chunk(self, batch: dict, chunk_id: int) -> None:
        """
        Writes given chunk with its respective id to the output directory.

        Args:
            batch (dict): Column-oriented batch from the streaming dataset.
            chunk_id (int): Id used for the output chunk filename.

        Returns:
            None.
        """
        path = self.output_path / f"chunk_{chunk_id:05d}.parquet"
        tmp = path.with_suffix(".tmp")

        filtered = {k: batch[k] for k in DataConfig.keep_columns}
        pq.write_table(pa.table(filtered), tmp, compression="zstd")
        os.replace(tmp, path)

    def _target_reached(self) -> bool:
        """
        Checks whether the configured stopping condition has been reached.

        Returns:
            bool: True if the doc limit (when set) or the token target has been hit.
        """
        if self.limit is not None:
            return self.progress["docs"] >= self.limit
        return self.progress["tokens"] >= self.target_tokens

    def _progress_metric(self) -> tuple[int, int, str]:
        """
        Returns the (current, target, unit) for whichever stopping condition
        is active: doc count when testing with `limit`, else token count.

        Returns:
            tuple[int, int, str]: Current value, target value, unit label.
        """
        if self.limit is not None:
            return self.progress["docs"], self.limit, "docs"
        return self.progress["tokens"], self.target_tokens, "tokens"

    def run(self) -> None:
        """
        Runs the downloader pipeline, which handles the download and save in proper format of the dataset.

        Returns:
            None.
        """
        if self._target_reached():
            print("Target already reached, nothing to do.")
            return

        start_time = time.monotonic()
        start_value, _, _ = self._progress_metric()

        chunk_iter = enumerate(self.dataset.iter(batch_size=self.docs_per_chunk), start=self.progress["chunks"])
        for chunk_id, batch in chunk_iter:
            print(f"Starting chunk {chunk_id:05d}... (elapsed {_format_duration(time.monotonic() - start_time)})")
            self._write_chunk(batch, chunk_id)

            self.progress["chunks"] += 1
            self.progress["docs"] += len(batch["id"])
            self.progress["tokens"] += sum(batch["token_count"])
            self.progress["state"] = self.dataset.state_dict()
            self._save_progress()

            current, target, unit = self._progress_metric()
            elapsed = time.monotonic() - start_time
            rate = (current - start_value) / elapsed if elapsed > 0 else 0.0
            eta = (target - current) / rate if rate > 0 else None
            completed = current / target

            print(
                f"Chunk {self.progress['chunks']:05d} | "
                f"{self.progress['docs']} docs | "
                f"{self.progress['tokens'] / 1e9:.2f}B tokens | {completed:.1%} | "
                f"{rate:.1f} {unit}/s | elapsed {_format_duration(elapsed)} | "
                f"eta {_format_duration(eta) if eta is not None else 'unknown'}"
            )

            if self._target_reached():
                break

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream FineWeb-Edu and write it to local parquet chunks.")
    parser.add_argument("--output", type=Path, default=DataConfig.output_path, help="Directory for chunks and progress.json.")
    parser.add_argument("--target", type=int, default=DataConfig.target_tokens, help="Token budget to stop at.")
    parser.add_argument("--batch-size", type=int, default=DataConfig.docs_per_chunk, help="Docs per streamed batch / chunk file.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many docs (testing only, overrides --target).")
    return parser.parse_args()

def main() -> None:
    """
    Runs the downloader functionality with CLI commands.
    """
    args = parse_args()
    downloader = Downloader(
        output_path=args.output,
        target_tokens=args.target,
        docs_per_chunk=args.batch_size,
        limit=args.limit,
    )
    downloader.run()

if __name__ == "__main__":
    main()
