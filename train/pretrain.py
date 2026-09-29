import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from huggingface_hub import hf_hub_download, upload_file

from config import GlobalConfig, TokenizerConfig, TrainConfig, MODEL_PRESETS
from data.progress import format_duration
from model.model import TransformerLM, save_checkpoint, load_checkpoint
from model.helper.data_loader import DataLoader
from model.helper.precision import detect_precision
from model.optim import build_optimizers
from model.scheduler import build_scheduler
from tokenizer.tokenizer_hf import build_hf_tokenizer

FIXED_PROMPTS = [
    "The history of the Roman Empire began",
    "def fibonacci(n):",
    "The capital of France is",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyForge pretraining trainer")
    parser.add_argument("--preset", choices=list(MODEL_PRESETS), default="xs")
    parser.add_argument("--optim", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--sched", choices=["wsd", "cosine"], default="wsd")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--aux-lr", type=float, default=None)
    parser.add_argument("--total-tokens", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=TrainConfig.val_interval)
    parser.add_argument("--ckpt-interval", type=int, default=TrainConfig.ckpt_interval)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="tinyforge")
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--overfit-batch", action="store_true")
    parser.add_argument("--seed", type=int, default=GlobalConfig.seed)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    return parser.parse_args()

def resolve_checkpoint_source(resume_from: str) -> Path:
    """Local path if it exists on disk, else treated as '<hf_repo_id>/<filename>'."""
    local_path = Path(resume_from)
    if local_path.exists():
        return local_path
    repo_id, filename = resume_from.rsplit("/", 1)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))

def push_checkpoint_to_hub(ckpt_path: Path, hf_repo_id: str | None) -> None:
    """Best-effort push; never hard-fails a local run with no HF auth configured."""
    if not hf_repo_id:
        return
    try:
        upload_file(path_or_fileobj=str(ckpt_path), path_in_repo=ckpt_path.name, repo_id=hf_repo_id)
    except Exception as e:
        print(f"Warning: failed to push checkpoint to HF Hub: {e}")

@torch.no_grad()
def evaluate(model, loader: DataLoader, batch_size: int, device, amp_dtype, amp_enabled: bool) -> float:
    model.eval()
    losses = []
    for inputs, targets in loader.val_batches(batch_size):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

@torch.no_grad()
def sample_fixed_prompts(model, tokenizer, device, context_length: int, eot_id: int, max_new_tokens: int = 40) -> list[list[str]]:
    """Naive greedy decode, no KV cache (that's a 2.d item)."""
    model.eval()
    rows = []
    for prompt in FIXED_PROMPTS:
        ids = tokenizer.encode(prompt).ids
        for _ in range(max_new_tokens):
            window = ids[-context_length:]
            inputs = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(inputs)
            next_id = int(logits[0, -1].argmax())
            ids.append(next_id)
            if next_id == eot_id:
                break
        rows.append([prompt, tokenizer.decode(ids)])
    model.train()
    return rows

def train_loop(
    model, optimizers, schedulers, scaler, loader: DataLoader, tokenizer, args,
    preset, device, amp_dtype, amp_enabled: bool, start_step: int, total_steps: int,
    tokens_per_step: int, wandb_run_id: str,
) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = TokenizerConfig.vocab_size
    cached_batch = None
    start_time = time.monotonic()

    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)

    for step in range(start_step, total_steps):
        step_start = time.monotonic()
        total_loss = 0.0

        for _ in range(args.grad_accum_steps):
            if args.overfit_batch and cached_batch is not None:
                inputs, targets = cached_batch
            else:
                inputs, targets = loader.sample_batch(args.micro_batch_size)
                if args.overfit_batch and cached_batch is None:
                    cached_batch = (inputs, targets)

            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(inputs)
                loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

            total_loss += loss.item()
            loss_to_backward = loss / args.grad_accum_steps
            if scaler is not None:
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

        # Read the LR that's about to be used for this step's update, before sched.step()
        # advances it to next step's value -- logging it after would show next step's LR.
        current_lr = next(iter(optimizers.values())).param_groups[0]["lr"]

        if scaler is not None:
            for opt in optimizers.values():
                scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainConfig.grad_clip)
        if scaler is not None:
            for opt in optimizers.values():
                scaler.step(opt)
            scaler.update()
        else:
            for opt in optimizers.values():
                opt.step()
        for sched in schedulers.values():
            sched.step()
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        tokens_seen = (step + 1) * tokens_per_step
        step_elapsed = time.monotonic() - step_start
        tokens_per_sec = tokens_per_step / step_elapsed if step_elapsed > 0 else 0.0
        peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        train_loss = total_loss / args.grad_accum_steps

        wandb.log({
            "tokens_seen": tokens_seen,
            "train/loss": train_loss,
            "train/lr": current_lr,
            "train/tokens_per_sec": tokens_per_sec,
            "train/peak_mem_bytes": peak_mem,
        })
        print(f"step {step:6d} | tokens {tokens_seen:,} | loss {train_loss:.4f} | lr {current_lr:.2e} "
              f"| tok/s {tokens_per_sec:,.0f} | elapsed {format_duration(time.monotonic() - start_time)}")

        is_last_step = step == total_steps - 1
        if (step + 1) % args.val_interval == 0 or is_last_step:
            val_loss = evaluate(model, loader, args.micro_batch_size, device, amp_dtype, amp_enabled)
            wandb.log({"tokens_seen": tokens_seen, "val/loss": val_loss})
            print(f"  val loss {val_loss:.4f}")

        if (step + 1) % args.ckpt_interval == 0 or is_last_step:
            ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
            save_checkpoint(model, optimizers, step, ckpt_path, scaler=scaler,
                             data_loader_rng_state=loader.state_dict(), wandb_run_id=wandb_run_id)
            push_checkpoint_to_hub(ckpt_path, args.hf_repo_id)

            rows = sample_fixed_prompts(model, tokenizer, device, preset.context_length, TokenizerConfig.eot_id)
            wandb.log({"tokens_seen": tokens_seen, "samples": wandb.Table(columns=["prompt", "completion"], data=rows)})

def main() -> None:
    args = parse_args()
    preset = MODEL_PRESETS[args.preset]
    torch.manual_seed(args.seed)

    amp_dtype, needs_scaler = detect_precision()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = amp_dtype != torch.float32

    # Model weights always start in fp32; autocast handles bf16/fp16 compute during
    # the forward pass, which is required for fp16+GradScaler's numerics to be stable.
    model = TransformerLM(
        vocab_size=TokenizerConfig.vocab_size,
        context_length=preset.context_length,
        num_layers=preset.n_layer,
        d_model=preset.d_model,
        num_heads=preset.n_head,
        d_ff=preset.d_ff,
        max_seq_len=preset.context_length,
        theta=preset.theta,
        device=device,
    ).to(device)

    lr = args.lr if args.lr is not None else getattr(TrainConfig, f"peak_lr_{args.optim}")
    optimizers = build_optimizers(model, args.optim, lr=lr, aux_lr=args.aux_lr,
                                   weight_decay=TrainConfig.weight_decay, betas=TrainConfig.betas)
    scaler = torch.amp.GradScaler(device="cuda") if needs_scaler else None

    loader = DataLoader(shards_path=TokenizerConfig.shards_path, context_length=preset.context_length,
                         seed=args.seed, num_val_windows=TrainConfig.num_val_windows)

    tokens_per_step = args.micro_batch_size * args.grad_accum_steps * preset.context_length
    total_steps = max(1, args.total_tokens // tokens_per_step)

    start_step = 0
    wandb_run_id = None
    if args.resume_from:
        ckpt_path = resolve_checkpoint_source(args.resume_from)
        bookkeeping = load_checkpoint(ckpt_path, model, optimizers, scaler=scaler)
        start_step = bookkeeping["step"] + 1
        loader.load_state_dict(bookkeeping["data_loader_rng_state"])
        wandb_run_id = bookkeeping["wandb_run_id"]

    # PyTorch's LambdaLR always applies lr_lambda(last_epoch + 1) at construction (a
    # plain .step() bump), including for the last_epoch=-1 default (-1 + 1 = 0). So to
    # make the upcoming step `start_step` use lr_lambda(start_step), as it would have in
    # an uninterrupted run, we must pass last_epoch=start_step-1 -- which also happens to
    # equal -1 for a fresh run (start_step=0), giving the correct default behavior for
    # free. A resumed run's start_step-1 >= 0 requires 'initial_lr' already present in
    # the optimizer's param groups, which load_checkpoint's restored state provides.
    schedulers = {
        name: build_scheduler(args.sched, opt, total_steps, warmup_frac=TrainConfig.warmup_frac,
                               decay_frac=TrainConfig.wsd_decay_frac, min_lr_frac=TrainConfig.cosine_min_lr_frac,
                               last_epoch=start_step - 1)
        for name, opt in optimizers.items()
    }

    wandb.init(project=args.wandb_project, id=wandb_run_id, resume="allow" if wandb_run_id else None, config=vars(args))
    wandb_run_id = wandb.run.id
    wandb.define_metric("tokens_seen")
    wandb.define_metric("*", step_metric="tokens_seen")

    tokenizer = build_hf_tokenizer()

    train_loop(model, optimizers, schedulers, scaler, loader, tokenizer, args, preset, device,
               amp_dtype, amp_enabled, start_step, total_steps, tokens_per_step, wandb_run_id)

if __name__ == "__main__":
    main()
