import argparse
import json
import math
from pathlib import Path

import torch

import lm_eval
from lm_eval.api.model import LM

from config import TokenizerConfig, MODEL_PRESETS
from model.model import TransformerLM
from model.helper.checkpoint_source import resolve_checkpoint_source
from model.helper.data_loader import DataLoader
from model.helper.precision import detect_precision
from tokenizer.tokenizer_hf import build_hf_tokenizer
from train.pretrain import evaluate

def load_model_for_eval(checkpoint_ref: str, preset_name: str, device: torch.device) -> TransformerLM:
    """
    Builds a TransformerLM from a MODEL_PRESETS entry and loads weights only
    (no optimizer/scaler/RNG state -- this is for inference, not resuming a run).

    Args:
        checkpoint_ref: Local path or '<hf_repo>/<...>/<file>.pt' reference.
        preset_name: Key into MODEL_PRESETS (xs/s/m/base60m) -- checkpoints don't
            currently embed their own architecture, so this must match how the
            checkpoint was trained.
        device: Device to load the model onto.
    Returns:
        TransformerLM in eval() mode.
    """
    preset = MODEL_PRESETS[preset_name]
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

    ckpt_path = resolve_checkpoint_source(checkpoint_ref)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def compute_val_perplexity(model, device: torch.device, preset, shards_path: Path, batch_size: int = 16) -> dict:
    """
    Reuses train.pretrain.evaluate()
    """
    amp_dtype, _ = detect_precision()
    amp_enabled = amp_dtype != torch.float32
    loader = DataLoader(shards_path=shards_path, context_length=preset.context_length)
    val_loss = evaluate(model, loader, batch_size, device, amp_dtype, amp_enabled)
    return {"val_loss": val_loss, "val_perplexity": math.exp(val_loss)}

class TinyForgeLM(LM):
    """
    lm-eval adapter around TransformerLM + the project's own BPE tokenizer.
    """
    def __init__(self, model: TransformerLM, tokenizer, device: torch.device, context_length: int, eot_id: int):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self.context_length = context_length
        self.eot_id = eot_id
        self.amp_dtype, _ = detect_precision()
        self.amp_enabled = self.amp_dtype != torch.float32

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids if text else []

    @torch.no_grad()
    def loglikelihood(self, requests: list) -> list[tuple[float, bool]]:
        results = []
        for req in requests:
            context, continuation = req.args
            continuation_ids = self._encode(continuation)
            if not continuation_ids:
                results.append((0.0, True))
                continue

            # Always score the continuation in full
            if len(continuation_ids) >= self.context_length:
                continuation_ids = continuation_ids[-(self.context_length - 1):]
            max_context_len = self.context_length - len(continuation_ids)
            context_ids = self._encode(context)
            context_ids = context_ids[-max_context_len:] if context_ids else []
            if not context_ids:
                context_ids = [self.eot_id]

            ids = context_ids + continuation_ids
            inputs = torch.tensor([ids[:-1]], dtype=torch.long, device=self._device)
            targets = torch.tensor(ids[1:], dtype=torch.long, device=self._device)

            with torch.autocast(device_type=self._device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                logits = self.model(inputs)[0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            cont_len = len(continuation_ids)
            cont_log_probs = log_probs[-cont_len:]
            cont_targets = targets[-cont_len:]

            token_logprobs = cont_log_probs.gather(-1, cont_targets.unsqueeze(-1)).squeeze(-1)
            is_greedy = bool(torch.equal(cont_log_probs.argmax(dim=-1), cont_targets))

            results.append((float(token_logprobs.sum().item()), is_greedy))
        return results

    def loglikelihood_rolling(self, requests: list) -> list[float]:
        raise NotImplementedError("Not needed for hellaswag/arc_easy (multiple-choice loglikelihood tasks).")

    def generate_until(self, requests: list) -> list[str]:
        raise NotImplementedError("Not needed for hellaswag/arc_easy (multiple-choice loglikelihood tasks).")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyForge benchmark evaluation: val perplexity + lm-eval tasks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Local path or '<hf_repo>/<...>/<file>.pt'")
    parser.add_argument("--preset", choices=list(MODEL_PRESETS), required=True)
    parser.add_argument("--tasks", type=str, default="hellaswag,arc_easy")
    parser.add_argument("--limit", type=int, default=None, help="Cap examples per task, for a quick smoke run")
    parser.add_argument("--shards-path", type=str, default=str(TokenizerConfig.shards_path))
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Batch size for the val-perplexity pass only")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = MODEL_PRESETS[args.preset]

    model = load_model_for_eval(args.checkpoint, args.preset, device)
    tokenizer = build_hf_tokenizer()

    print("Computing val perplexity...")
    perplexity_report = compute_val_perplexity(model, device, preset, Path(args.shards_path), args.eval_batch_size)
    print(f"  val_loss={perplexity_report['val_loss']:.4f} val_perplexity={perplexity_report['val_perplexity']:.4f}")

    adapter = TinyForgeLM(model, tokenizer, device, preset.context_length, TokenizerConfig.eot_id)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"Running lm-eval tasks: {tasks} (limit={args.limit})...")
    lm_eval_results = lm_eval.simple_evaluate(model=adapter, tasks=tasks, limit=args.limit)

    report = {
        **perplexity_report,
        "lm_eval_results": lm_eval_results["results"] if lm_eval_results else None,
        "checkpoint": args.checkpoint,
        "preset": args.preset,
        "tasks": tasks,
        "limit": args.limit,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {output_path}")

if __name__ == "__main__":
    main()
