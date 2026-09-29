import math
from typing import Literal

import torch

def _wsd_multiplier(step: int, total_steps: int, warmup_frac: float, decay_frac: float) -> float:
    """
    Linear warmup -> flat at 1.0 -> linear decay to 0 over the last decay_frac.
    """
    warmup_steps = max(1, int(total_steps * warmup_frac))
    decay_steps = max(1, int(total_steps * decay_frac))
    decay_start = total_steps - decay_steps

    if step < warmup_steps:
        return step / warmup_steps
    if step < decay_start:
        return 1.0
    return max(0.0, 1.0 - (step - decay_start) / decay_steps)

def _cosine_multiplier(step: int, total_steps: int, warmup_frac: float, min_lr_frac: float) -> float:
    """
    Same warmup, then cosine decay from 1.0 down to min_lr_frac (not to 0).
    """
    warmup_steps = max(1, int(total_steps * warmup_frac))

    if step < warmup_steps:
        return step / warmup_steps

    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_frac + (1 - min_lr_frac) * cosine

def build_scheduler(
    sched_name: Literal["wsd", "cosine"],
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_frac: float = 0.01,
    decay_frac: float = 0.175,
    min_lr_frac: float = 0.10,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Builds the LR schedule behind the --sched flag, as a step -> multiplier function
    fed to LambdaLR.

    `last_epoch` is what makes resume exact: LambdaLR derives the current LR from
    `last_epoch` at construction time, so passing last_epoch=step-1 reproduces the LR
    on resume without needing separate scheduler state in the checkpoint.

    Args:
        sched_name: "wsd" or "cosine".
        optimizer: Optimizer to schedule.
        total_steps: Total number of optimizer steps in the run.
        warmup_frac: Fraction of total_steps spent warming up.
        decay_frac: (wsd only) fraction of total_steps spent decaying at the end.
        min_lr_frac: (cosine only) LR floor as a fraction of peak.
        last_epoch: Step to resume the schedule from (-1 = start fresh).
    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """
    if sched_name == "wsd":
        lr_lambda = lambda step: _wsd_multiplier(step, total_steps, warmup_frac, decay_frac)
    elif sched_name == "cosine":
        lr_lambda = lambda step: _cosine_multiplier(step, total_steps, warmup_frac, min_lr_frac)
    else:
        raise ValueError(f"Unknown schedule: {sched_name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

if __name__ == "__main__":
    # Sanity check per the guide: "Plot both over a dummy 1,000 steps before trusting them."
    total_steps = 1000
    for step in range(0, total_steps, 100):
        wsd = _wsd_multiplier(step, total_steps, warmup_frac=0.01, decay_frac=0.175)
        cosine = _cosine_multiplier(step, total_steps, warmup_frac=0.01, min_lr_frac=0.10)
        print(f"step={step:4d} | wsd={wsd:.3f} | cosine={cosine:.3f}")
