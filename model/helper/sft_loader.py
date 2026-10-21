import json
import random
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from config import SFTConfig, TokenizerConfig
from tokenizer.chat_template import encode_conversation

def resolve_sft_file(filename: str) -> Path:
    """
    Local file under SFTConfig.output_path if present (made by
    train/prepare_sft_data.py), else downloaded from the SFT folder of the HF
    dataset repo -- so a fresh Colab session needs nothing but HF access.
    """
    local_path = SFTConfig.output_path / filename
    if local_path.exists():
        return local_path
    return Path(hf_hub_download(repo_id=SFTConfig.data_repo, repo_type="dataset",
                                filename=f"{SFTConfig.hub_path_in_repo}/{filename}"))

class SFTDataset:
    def __init__(self, path: Path, tokenizer, seed: int, limit: int | None = None):
        """
        Loads one SFT JSONL split and tokenizes every conversation up front through
        the shared chat template, so training sees exactly the ids the template
        gates were checked on.

        Args:
            path (Path): JSONL from train/prepare_sft_data.py.
            tokenizer: HF tokenizer from build_hf_tokenizer().
            seed (int): Seed for the per-epoch shuffle.
            limit (int | None): Keep only the first `limit` conversations (smoke runs).
        """
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = rows[:limit] if limit else rows

        self.ids, self.train_masks = [], []
        for row in tqdm(rows, desc=f"tokenizing {path.name}"):
            ids, train_mask = encode_conversation(row["messages"], tokenizer)
            self.ids.append(np.array(ids, dtype=np.uint16))
            self.train_masks.append(np.array(train_mask, dtype=bool))

        self.lengths = [len(ids) for ids in self.ids]
        self.seed = seed

    def __len__(self) -> int:
        return len(self.ids)

    def epoch_steps(self, epoch: int, micro_batch_size: int, grad_accum_steps: int, bucket_multiplier: int = 50) -> list[list[list[int]]]:
        """
        Plans one epoch: shuffle, then sort by length inside chunks of
        micro_batch_size * bucket_multiplier conversations so each micro-batch pads
        to similar lengths, then shuffle the micro-batches so steps don't go from
        short to long. Seeded by (seed, epoch), so the plan is reproducible and a
        run resumed at an epoch boundary sees the same order.

        Args:
            epoch (int): Epoch index.
            micro_batch_size (int): Conversations per micro-batch.
            grad_accum_steps (int): Micro-batches per optimizer step.
            bucket_multiplier (int): Chunk size for length sorting, in micro-batches.
        Returns:
            list[list[list[int]]]: steps -> micro-batches -> conversation indices.
        """
        rng = random.Random(self.seed * 1000 + epoch)
        order = list(range(len(self)))
        rng.shuffle(order)

        chunk_size = micro_batch_size * bucket_multiplier
        micro_batches = []
        for start in range(0, len(order), chunk_size):
            chunk = sorted(order[start:start + chunk_size], key=lambda i: self.lengths[i])
            micro_batches.extend(chunk[j:j + micro_batch_size] for j in range(0, len(chunk), micro_batch_size))
        rng.shuffle(micro_batches)

        return [micro_batches[s:s + grad_accum_steps] for s in range(0, len(micro_batches), grad_accum_steps)]

    def eval_batches(self, micro_batch_size: int) -> list[list[int]]:
        """
        Fixed, length-sorted micro-batches over the whole split, for held-out loss.
        """
        order = sorted(range(len(self)), key=lambda i: self.lengths[i])
        return [order[j:j + micro_batch_size] for j in range(0, len(order), micro_batch_size)]

    def collate(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Right-pads a micro-batch. inputs = ids[:-1] padded with the EOT id; labels =
        ids[1:] where the next token is a training target, else ignore_index (also
        on padding). Right padding needs no attention mask: under the causal mask,
        real tokens never attend to the padding after them.

        Args:
            indices (list[int]): Conversation indices.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (inputs, labels), each (batch, max_len - 1), long.
        """
        max_len = max(self.lengths[i] for i in indices) - 1
        inputs = np.full((len(indices), max_len), TokenizerConfig.eot_id, dtype=np.int64)
        labels = np.full((len(indices), max_len), SFTConfig.ignore_index, dtype=np.int64)

        for row, i in enumerate(indices):
            ids, train_mask = self.ids[i].astype(np.int64), self.train_masks[i]
            n = len(ids) - 1
            inputs[row, :n] = ids[:-1]
            labels[row, :n] = np.where(train_mask[1:], ids[1:], SFTConfig.ignore_index)

        return torch.from_numpy(inputs), torch.from_numpy(labels)
