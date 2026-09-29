import math
import os
from typing import BinaryIO, IO, Union

import torch
from torch import nn

from .components.embedding import Embedding
from .components.linear import Linear
from .components.rsm_norm import RMSNorm
from .components.transformer_block import TransformerBlock

PathOrFile = Union[str, os.PathLike, BinaryIO, IO[bytes]]

def save_checkpoint(
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    step: int,
    out: PathOrFile,
    scaler: "torch.amp.GradScaler | None" = None,
    data_loader_rng_state: dict | None = None,
    wandb_run_id: str | None = None,
) -> None:
    """
    Save a training checkpoint containing model/optimizers state, scaler,
    step, data-loader RNG state and W&B run id.

    Args:
        model: torch.nn.Module
        optimizers: Named optimizers (e.g. {"adamw": ...} or {"adamw": ..., "muon": ...}).
        step: Current training step.
        out: File path or a binary file-like object.
        scaler: GradScaler in use for fp16 training, if any.
        data_loader_rng_state: Opaque state dict from the data loader.
        wandb_run_id: W&B run id, for resuming the same run.
    """
    obj = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": {name: opt.state_dict() for name, opt in optimizers.items()},
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "data_loader_rng_state": data_loader_rng_state,
        "wandb_run_id": wandb_run_id,
    }
    torch.save(obj, out)

def load_checkpoint(
    src: PathOrFile,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    scaler: "torch.amp.GradScaler | None" = None,
) -> dict:
    """
    Load a training checkpoint and restore model/optimizers/scaler state in place.

    Args:
        src: File path or a binary file-like object.
        model: torch.nn.Module to restore into.
        optimizers: Named optimizers to restore into (same names used at save time).
        scaler: GradScaler to restore into, if training in fp16.

    Returns:
        dict with the remaining bookkeeping fields: "step", "data_loader_rng_state",
        "wandb_run_id" — left for the caller to apply, since this function has no
        business knowing about the data loader or W&B.
    """
    ckpt = torch.load(src, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dict.")

    if "model_state_dict" not in ckpt or "optimizer_state_dicts" not in ckpt or "step" not in ckpt:
        raise KeyError("Checkpoint dict missing required keys.")

    model.load_state_dict(ckpt["model_state_dict"])
    for name, opt in optimizers.items():
        opt.load_state_dict(ckpt["optimizer_state_dicts"][name])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return {
        "step": int(ckpt["step"]),
        "data_loader_rng_state": ckpt.get("data_loader_rng_state"),
        "wandb_run_id": ckpt.get("wandb_run_id"),
    }

class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        """
        Initializes the Transformer language model: embedding, blocks, final norm and head.

        Args:
            vocab_size (int): Size of the vocabulary.
            context_length (int): Maximum number of tokens accepted per forward pass.
            num_layers (int): Number of stacked Transformer blocks.
            d_model (int): Model hidden dimension.
            num_heads (int): Number of attention heads per block.
            d_ff (int): Hidden dimension of each block's feed-forward network.
            max_seq_len (int): Maximum sequence length supported by RoPE.
            theta (float): Θ value for RoPE.
            device (torch.device | None): Device to store the parameters on.
            dtype (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Setup embedding matrix
        self.embedding_matrix = Embedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.d_model,
                                          device=device,
                                          dtype=dtype)
        # Setup transformer blocks (based on num_layers)
        self.blocks = nn.ModuleList([TransformerBlock(d_model=self.d_model,
                                                      num_heads=num_heads,
                                                      d_ff=self.d_ff,
                                                      max_seq_len=self.max_seq_len,
                                                      theta=theta,
                                                      device=device,
                                                      dtype=dtype) for _ in range(self.num_layers)])
        # Setup RMSNorm
        self.rmsnorm = RMSNorm(d_model=self.d_model,
                               device=device,
                               dtype=dtype)
        # Setup Linear layer
        self.head = Linear(in_features=self.d_model,
                           out_features=self.vocab_size,
                           device=device,
                           dtype=dtype)

        # Weight tying: head and embedding share the same (vocab_size, d_model) matrix
        self.head.W = self.embedding_matrix.embedding_matrix

        self._rescale_residual_projections()
        self._report_param_counts()

    def _rescale_residual_projections(self) -> None:
        """
        Re-draws the attention output projection and FFN down-projection of every
        block from a truncated normal with std shrunk by 1/sqrt(2*num_layers), so the
        residual stream's variance doesn't grow with depth (GPT-2-style init).
        """
        new_std = 0.02 / math.sqrt(2 * self.num_layers)
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.endswith("o_proj.W") or name.endswith("ffn.w2.W"):
                    nn.init.trunc_normal_(param, mean=0.0, std=new_std, a=-3 * new_std, b=3 * new_std)

    def _report_param_counts(self) -> None:
        """Prints embedding vs non-embedding param counts (tied embedding/head matrix
        counted once, since self.parameters() dedups shared tensors by identity)."""
        embedding_params = self.embedding_matrix.embedding_matrix.numel()
        total_params = sum(p.numel() for p in self.parameters())
        non_embedding_params = total_params - embedding_params
        print(f"Param count | embedding: {embedding_params:,} | non-embedding: {non_embedding_params:,} | total: {total_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the full forward pass and returns next-token logits.

        Args:
            x (torch.Tensor): Input token IDs of shape (batch, seq_len).
        Returns:
            torch.Tensor: Logits of shape (batch, seq_len, vocab_size).
        """
        batch, seq_len = x.shape
        # Check that sequence fits in context
        if seq_len > self.context_length:
            raise ValueError(f"seq_len={seq_len} exceeds context_length={self.context_length}")

        # Embed tokens
        x = self.embedding_matrix(x)
        # Generate token positions for RoPE
        idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        # Apply all transformer blocks
        for block in self.blocks:
            x = block(x, idx)
        # Apply final norm and head
        x = self.rmsnorm(x)
        return self.head(x)