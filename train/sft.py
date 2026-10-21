import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from huggingface_hub import create_repo

from config import GlobalConfig, SFTConfig, TokenizerConfig, TrainConfig, MODEL_PRESETS
from data.progress import format_duration
from model.generate import greedy_generate_ids
from model.helper.checkpoint_source import resolve_checkpoint_source
from model.helper.precision import detect_precision
from model.helper.sft_loader import SFTDataset, resolve_sft_file
from model.model import TransformerLM, save_checkpoint, load_checkpoint
from model.optim import build_optimizers
from model.scheduler import build_scheduler
from tokenizer.chat_template import END_OF_TURN, encode_prompt
from tokenizer.tokenizer_hf import build_hf_tokenizer
from train.pretrain import push_checkpoint_to_hub

SAMPLE_PROMPTS = [
    "What is photosynthesis?",
    "Write a Python function that checks whether a number is prime.",
    "Give me three tips for studying for an exam.",
]

RUN_FILES = {
    "sft-50k": SFTConfig.train_main_file,
    "sft-2k": SFTConfig.train_ablation_file,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyForge SFT trainer")
    parser.add_argument("--run-name", choices=list(RUN_FILES), required=True)
    parser.add_argument("--preset", choices=list(MODEL_PRESETS), default=SFTConfig.preset)
    parser.add_argument("--base-checkpoint", type=str, default=SFTConfig.base_checkpoint,
                        help="Local path or '<hf_repo>/<...>/<file>.pt'; 'none' = random init (gates only)")
    parser.add_argument("--lr", type=float, default=SFTConfig.lr)
    parser.add_argument("--epochs", type=int, default=SFTConfig.epochs)
    parser.add_argument("--micro-batch-size", type=int, default=SFTConfig.micro_batch_size)
    parser.add_argument("--grad-accum-steps", type=int, default=SFTConfig.grad_accum_steps)
    parser.add_argument("--eval-interval", type=int, default=SFTConfig.eval_interval)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop early (smoke runs); the LR schedule is compressed to fit")
    parser.add_argument("--train-limit", type=int, default=None, help="Use only the first N train conversations (smoke runs)")
    parser.add_argument("--heldout-limit", type=int, default=None, help="Use only the first N held-out conversations (smoke runs)")
    parser.add_argument("--overfit-batch", action="store_true", help="Repeat the first step's batch every step (gate)")
    parser.add_argument("--resume-from", type=str, default=None, help="An end-of-epoch checkpoint")
    parser.add_argument("--seed", type=int, default=GlobalConfig.seed)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--wandb-project", type=str, default="tinyforge")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--hf-repo-id", type=str, default=None, help=f"e.g. {SFTConfig.hub_model_repo}; created private if missing")
    return parser.parse_args()

def build_model(preset_name: str, device: torch.device) -> TransformerLM:
    preset = MODEL_PRESETS[preset_name]
    return TransformerLM(
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

@torch.no_grad()
def init_chat_token_embeddings(model: TransformerLM) -> None:
    """
    Sets the <|assistant|>/<|user|>/<|endofturn|> rows of the tied embedding/head
    matrix to the mean of the regular-token rows. These ids never occurred in
    pretraining, so their rows only ever got pushed toward low logits; the mean
    row starts them at a neutral point. <|endoftext|> was trained (document
    separator) and is left alone, but excluded from the mean.
    """
    special_ids = [256 + i for i in range(len(TokenizerConfig.special_tokens))]
    chat_ids = special_ids[1:]
    weight = model.embedding_matrix.embedding_matrix
    regular = torch.ones(weight.size(0), dtype=torch.bool, device=weight.device)
    regular[special_ids] = False
    weight[chat_ids] = weight[regular].mean(dim=0)

def masked_loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Sum (not mean) of cross-entropy over target tokens. The caller divides by the
    target count of the whole optimizer step, so every assistant token weighs the
    same regardless of which micro-batch it landed in.
    """
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                           ignore_index=SFTConfig.ignore_index, reduction="sum")

def count_targets(labels: torch.Tensor) -> int:
    return int((labels != SFTConfig.ignore_index).sum())

@torch.no_grad()
def evaluate_heldout(model, dataset: SFTDataset, micro_batch_size: int, device, amp_dtype, amp_enabled: bool) -> float:
    """
    Mean cross-entropy over every assistant target token in the held-out split.
    """
    model.eval()
    loss_sum, targets = 0.0, 0
    for indices in dataset.eval_batches(micro_batch_size):
        inputs, labels = (t.to(device) for t in dataset.collate(indices))
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss_sum += masked_loss_sum(model(inputs), labels).item()
        targets += count_targets(labels)
    model.train()
    return loss_sum / targets

@torch.no_grad()
def sample_chat(model, tokenizer, context_length: int, max_new_tokens: int = 64) -> list[list[str]]:
    model.eval()
    stop_ids = {tokenizer.token_to_id(END_OF_TURN), TokenizerConfig.eot_id}
    rows = []
    for prompt in SAMPLE_PROMPTS:
        prompt_ids = encode_prompt([{"role": "user", "content": prompt}], tokenizer)
        new_ids = greedy_generate_ids(model, prompt_ids, context_length, stop_ids, max_new_tokens)
        rows.append([prompt, tokenizer.decode(new_ids, skip_special_tokens=False)])
    model.train()
    return rows

def main() -> None:
    args = parse_args()
    preset = MODEL_PRESETS[args.preset]
    torch.manual_seed(args.seed)

    amp_dtype, needs_scaler = detect_precision()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = amp_dtype != torch.float32

    tokenizer = build_hf_tokenizer()
    train_set = SFTDataset(resolve_sft_file(RUN_FILES[args.run_name]), tokenizer, args.seed, limit=args.train_limit)
    heldout_set = SFTDataset(resolve_sft_file(SFTConfig.heldout_file), tokenizer, args.seed, limit=args.heldout_limit)

    model = build_model(args.preset, device)
    optimizers = build_optimizers(model, SFTConfig.optimizer, lr=args.lr,
                                   weight_decay=SFTConfig.weight_decay, betas=TrainConfig.betas)
    scaler = torch.amp.GradScaler(device="cuda") if needs_scaler else None

    steps_per_epoch = len(train_set.epoch_steps(0, args.micro_batch_size, args.grad_accum_steps))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    start_epoch, step = 0, 0
    wandb_run_id = None
    if args.resume_from:
        bookkeeping = load_checkpoint(resolve_checkpoint_source(args.resume_from), model, optimizers, scaler=scaler)
        progress = bookkeeping["data_loader_rng_state"] or {}
        if not progress.get("epoch_complete"):
            raise ValueError("Can only resume from an end-of-epoch checkpoint.")
        start_epoch, step = progress["epoch"] + 1, bookkeeping["step"] + 1
        wandb_run_id = bookkeeping["wandb_run_id"]
    elif args.base_checkpoint != "none":
        ckpt = torch.load(resolve_checkpoint_source(args.base_checkpoint), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        init_chat_token_embeddings(model)

    schedulers = {
        name: build_scheduler("wsd", opt, total_steps, warmup_frac=SFTConfig.warmup_frac,
                               decay_frac=SFTConfig.decay_frac, last_epoch=step - 1)
        for name, opt in optimizers.items()
    }

    wandb.init(project=args.wandb_project, name=args.run_name, id=wandb_run_id, mode=args.wandb_mode,
               resume="allow" if wandb_run_id else None, config=vars(args))
    wandb_run_id = wandb.run.id
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")

    if args.hf_repo_id:
        create_repo(args.hf_repo_id, private=True, exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.run_name}: {len(train_set):,} train / {len(heldout_set):,} held-out conversations, "
          f"{steps_per_epoch} steps/epoch, {total_steps} total steps")

    if step == 0:
        heldout_loss = evaluate_heldout(model, heldout_set, args.micro_batch_size, device, amp_dtype, amp_enabled)
        wandb.log({"step": 0, "heldout/loss": heldout_loss})
        print(f"step 0 (before SFT) | heldout loss {heldout_loss:.4f}")

    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)

    start_time = time.monotonic()
    trained_tokens_seen = 0
    cached_step = None
    for epoch in range(start_epoch, args.epochs):
        for micro_batches in train_set.epoch_steps(epoch, args.micro_batch_size, args.grad_accum_steps):
            if step >= total_steps:
                break
            step_start = time.monotonic()

            if args.overfit_batch:
                cached_step = cached_step or [train_set.collate(indices) for indices in micro_batches]
                batches = cached_step
            else:
                batches = [train_set.collate(indices) for indices in micro_batches]
            step_targets = sum(count_targets(labels) for _, labels in batches)

            loss_sum = 0.0
            for inputs, labels in batches:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    micro_loss_sum = masked_loss_sum(model(inputs), labels)
                loss_sum += micro_loss_sum.item()
                loss_to_backward = micro_loss_sum / step_targets
                if scaler is not None:
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

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

            step_tokens = sum(inputs.numel() for inputs, _ in batches)
            step_elapsed = time.monotonic() - step_start
            trained_tokens_seen += step_targets
            train_loss = loss_sum / step_targets
            wandb.log({
                "step": step + 1,
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/lr": current_lr,
                "train/trained_tokens_seen": trained_tokens_seen,
                "train/tokens_per_sec": step_tokens / step_elapsed if step_elapsed > 0 else 0.0,
                "train/peak_mem_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
            })
            print(f"epoch {epoch + 1} | step {step + 1:5d}/{total_steps} | loss {train_loss:.4f} | lr {current_lr:.2e} "
                  f"| tok/s {step_tokens / step_elapsed:,.0f} | elapsed {format_duration(time.monotonic() - start_time)}")
            step += 1

            if step % args.eval_interval == 0 and step < total_steps:
                heldout_loss = evaluate_heldout(model, heldout_set, args.micro_batch_size, device, amp_dtype, amp_enabled)
                wandb.log({"step": step, "heldout/loss": heldout_loss})
                print(f"  heldout loss {heldout_loss:.4f}")

        epoch_complete = step % steps_per_epoch == 0
        heldout_loss = evaluate_heldout(model, heldout_set, args.micro_batch_size, device, amp_dtype, amp_enabled)
        rows = sample_chat(model, tokenizer, preset.context_length)
        wandb.log({"step": step, "heldout/loss": heldout_loss, "heldout/epoch_end_loss": heldout_loss,
                   "samples": wandb.Table(columns=["prompt", "completion"], data=rows)})
        print(f"  end of {'epoch ' + str(epoch + 1) if epoch_complete else 'run'} | heldout loss {heldout_loss:.4f}")
        for prompt, completion in rows:
            print(f"    {prompt!r} -> {completion!r}")

        ckpt_name = f"epoch_{epoch + 1}.pt" if epoch_complete else f"step_{step:06d}.pt"
        ckpt_path = ckpt_dir / ckpt_name
        save_checkpoint(model, optimizers, step - 1, ckpt_path, scaler=scaler,
                        data_loader_rng_state={"epoch": epoch, "epoch_complete": epoch_complete},
                        wandb_run_id=wandb_run_id)
        push_checkpoint_to_hub(ckpt_path, args.hf_repo_id, args.run_name)

        if step >= total_steps:
            break

    wandb.finish()

if __name__ == "__main__":
    main()
