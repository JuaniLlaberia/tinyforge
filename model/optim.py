from typing import Literal

import torch
from torch import nn
from torch.optim import AdamW

# The 7 in-block 2D hidden weight matrices Muon is allowed to touch.
_HIDDEN_SUFFIXES = (
    "attn.q_proj.W", "attn.k_proj.W", "attn.v_proj.W", "attn.o_proj.W",
    "ffn.w1.W", "ffn.w2.W", "ffn.w3.W",
)

def _is_hidden_matrix(name: str) -> bool:
    return name.startswith("blocks.") and name.endswith(_HIDDEN_SUFFIXES)

def build_optimizers(
    model: nn.Module,
    optim_name: Literal["adamw", "muon"],
    lr: float,
    aux_lr: float | None = None,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> dict[str, torch.optim.Optimizer]:
    """
    Builds the optimizer(s) behind the --optim flag.

    Norm gains and the tied embedding/head matrix always go to AdamW with no weight
    decay, at `aux_lr` (defaults to `lr`). The 7 in-block hidden 2D weight matrices go
    to Muon (at `lr`) when optim_name == "muon", otherwise they're folded into the
    AdamW-with-decay bucket (also at `lr`). There are no bias terms anywhere in this
    architecture, so no bias carve-out is needed.

    Note: because head/embedding weights are tied by identity, `head.W` never appears
    as its own name in model.named_parameters() — only "embedding_matrix.embedding_matrix"
    needs to be matched.

    Args:
        model: The model whose parameters to optimize.
        optim_name: "adamw" or "muon".
        lr: Peak LR for the optimizer covering hidden matrices (or the sole optimizer
            when optim_name == "adamw").
        aux_lr: Peak LR for the always-AdamW slice (embeddings/head/norms). Defaults
            to `lr` when not given; kept separate since AdamW and Muon don't share a
            good LR.
        weight_decay: Weight decay applied to decayed params.
        betas: AdamW betas.
    Returns:
        dict[str, Optimizer]: {"adamw": ...} or {"adamw": ..., "muon": ...}.
    """
    aux_lr = lr if aux_lr is None else aux_lr

    no_decay, decay, hidden = [], [], []
    for name, param in model.named_parameters():
        if name.endswith(".g") or name == "embedding_matrix.embedding_matrix":
            no_decay.append(param)
        elif optim_name == "muon" and _is_hidden_matrix(name):
            hidden.append(param)
        else:
            decay.append(param)

    adamw_groups = [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": aux_lr},
    ]
    optimizers = {"adamw": AdamW(adamw_groups, betas=betas)}

    if optim_name == "muon":
        optimizers["muon"] = torch.optim.Muon(hidden, lr=lr, weight_decay=weight_decay, momentum=0.95, nesterov=True)

    return optimizers
