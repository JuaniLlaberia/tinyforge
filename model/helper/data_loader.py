import json
import random
from pathlib import Path

import numpy as np
import torch

from config import GlobalConfig, TokenizerConfig

class ShardSet:
    def __init__(self, shard_dir: Path, shard_files: list[str]):
        """
        Opens one memmap per shard file and builds a cumulative-offset index so a
        global token index can be mapped to (shard, local offset).

        Args:
            shard_dir (Path): Directory containing the shard files.
            shard_files (list[str]): Shard filenames, in order.
        """
        self.memmaps = [np.memmap(shard_dir / name, dtype=np.uint16, mode="r") for name in shard_files]
        self.lengths = [len(m) for m in self.memmaps]
        self.cumulative = np.cumsum([0] + self.lengths)
        self.total_tokens = int(self.cumulative[-1])

    def read_window(self, start: int, length: int) -> np.ndarray:
        """
        Reads `length` contiguous tokens starting at global offset `start`, looping
        across shard files if the window straddles one or more boundaries.

        Args:
            start (int): Global token offset to start reading from.
            length (int): Number of tokens to read.
        Returns:
            np.ndarray: uint16 array of shape (length,).
        """
        shard_idx = int(np.searchsorted(self.cumulative, start, side="right") - 1)
        local_start = start - int(self.cumulative[shard_idx])

        chunks = []
        remaining = length
        while remaining > 0:
            mm = self.memmaps[shard_idx]
            take = min(len(mm) - local_start, remaining)
            chunks.append(np.asarray(mm[local_start:local_start + take]))
            remaining -= take
            shard_idx += 1
            local_start = 0

        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

class DataLoader:
    def __init__(self,
                 shards_path: Path = TokenizerConfig.shards_path,
                 context_length: int = 1024,
                 seed: int = GlobalConfig.seed,
                 num_val_windows: int = 512):
        """
        Memmap-backed loader over the uint16 train/val shards. Train windows are drawn
        randomly (seeded, resumable); val windows are fixed and deterministic so val
        loss is comparable across runs.

        Args:
            shards_path (Path): Directory holding train/ and val/ shard subfolders;
                meta.json is expected at shards_path.parent / "meta.json".
            context_length (int): Number of input tokens per window (window is
                context_length + 1 tokens: inputs + shifted targets).
            seed (int): Seed for the train-sampling RNG.
            num_val_windows (int): Number of fixed, non-overlapping val windows.
        """
        self.window_len = context_length + 1

        meta = json.loads((shards_path.parent / "meta.json").read_text())
        self.train = ShardSet(shards_path / "train", sorted(meta["train_shards"]))
        self.val = ShardSet(shards_path / "val", sorted(meta["val_shards"]))

        self.rng = random.Random(seed)
        self.num_val_windows = min(num_val_windows, self.val.total_tokens // self.window_len)

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draws `batch_size` random windows from the concatenated train shards.

        Args:
            batch_size (int): Number of windows to draw.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (inputs, targets), each of shape
                (batch_size, context_length), dtype long.
        """
        max_start = self.train.total_tokens - self.window_len
        inputs, targets = [], []
        for _ in range(batch_size):
            start = self.rng.randrange(0, max_start + 1)
            window = self.train.read_window(start, self.window_len).astype(np.int64)
            inputs.append(window[:-1])
            targets.append(window[1:])
        return torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(targets))

    def val_batches(self, batch_size: int):
        """
        Yields fixed, non-overlapping val batches: window i starts at i*window_len,
        for i in range(num_val_windows). No RNG involved, so identical across runs.

        Args:
            batch_size (int): Number of windows per yielded batch.
        Yields:
            tuple[torch.Tensor, torch.Tensor]: (inputs, targets) pairs, as in sample_batch.
        """
        for batch_start in range(0, self.num_val_windows, batch_size):
            batch_end = min(batch_start + batch_size, self.num_val_windows)
            inputs, targets = [], []
            for i in range(batch_start, batch_end):
                window = self.val.read_window(i * self.window_len, self.window_len).astype(np.int64)
                inputs.append(window[:-1])
                targets.append(window[1:])
            yield torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(targets))

    def state_dict(self) -> dict:
        """
        Returns the train-sampling RNG state, for checkpointing.
        """
        return {"rng_state": self.rng.getstate()}

    def load_state_dict(self, state: dict | None) -> None:
        """
        Restores the train-sampling RNG state from a checkpoint, if present.
        """
        if state and state.get("rng_state") is not None:
            self.rng.setstate(state["rng_state"])
