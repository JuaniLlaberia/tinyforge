import argparse
from pathlib import Path

import torch
from huggingface_hub import HfApi

from config import SFTConfig, TokenizerConfig, MODEL_PRESETS
from model.helper.checkpoint_source import resolve_checkpoint_source
from tokenizer.chat_template import render
from train.sft import build_model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strip a training checkpoint to weights only and publish it with its model card")
    parser.add_argument("--checkpoint", type=str, required=True, help="Local path or '<hf_repo>/<...>/<file>.pt'")
    parser.add_argument("--preset", choices=list(MODEL_PRESETS), default=SFTConfig.preset)
    parser.add_argument("--run-name", type=str, required=True, help="e.g. sft-50k; stored as metadata")
    parser.add_argument("--repo-id", type=str, default=SFTConfig.hub_model_repo)
    parser.add_argument("--filename", type=str, default="tinyforge-60m-instruct.pt")
    parser.add_argument("--model-card", type=str, default="documentation/model_card_TinyForge-60M-Instruct.md")
    parser.add_argument("--out-dir", type=str, default="checkpoints/release")
    parser.add_argument("--dry-run", action="store_true", help="Write the weights-only file locally, don't upload")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    ckpt = torch.load(resolve_checkpoint_source(args.checkpoint), map_location="cpu")
    progress = ckpt.get("data_loader_rng_state") or {}
    if not progress.get("epoch_complete"):
        raise ValueError("Expected an end-of-epoch SFT checkpoint.")

    # Must load strictly into a fresh model of the stated preset before anything is published.
    model = build_model(args.preset, torch.device("cpu"))
    model.load_state_dict(ckpt["model_state_dict"])
    assert model.head.W is model.embedding_matrix.embedding_matrix

    # Keeps the "model_state_dict" key so load_model_for_eval reads it like any training
    # checkpoint; optimizer/scaler/RNG state is dropped. The tied embedding/head matrix
    # shares storage, so torch.save writes it once.
    release = {
        "model_state_dict": ckpt["model_state_dict"],
        "preset": args.preset,
        "run_name": args.run_name,
        "epoch": progress["epoch"] + 1,
        "step": ckpt["step"] + 1,
        "source_checkpoint": args.checkpoint,
        "base_checkpoint": SFTConfig.base_checkpoint,
        "special_tokens": {tok: 256 + i for i, tok in enumerate(TokenizerConfig.special_tokens)},
        "chat_format": render([{"role": "user", "content": "{user}"}, {"role": "assistant", "content": "{assistant}"}]),
    }

    out_path = Path(args.out_dir) / args.filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, out_path)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB): {args.run_name} epoch {release['epoch']}, "
          f"step {release['step']}")

    if args.dry_run:
        return

    api = HfApi()
    api.upload_file(path_or_fileobj=str(out_path), path_in_repo=args.filename, repo_id=args.repo_id)
    api.upload_file(path_or_fileobj=args.model_card, path_in_repo="README.md", repo_id=args.repo_id)
    print(f"Uploaded {args.filename} and README.md to {args.repo_id} (visibility unchanged)")

if __name__ == "__main__":
    main()
