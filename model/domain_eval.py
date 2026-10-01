import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from config import TokenizerConfig, MODEL_PRESETS
from model.evaluation import load_model_for_eval
from model.generate import greedy_generate
from tokenizer.tokenizer_hf import build_hf_tokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyForge domain-eval: run the fixed prompt set and save completions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Local path or '<hf_repo>/<...>/<file>.pt'")
    parser.add_argument("--preset", choices=list(MODEL_PRESETS), required=True)
    parser.add_argument("--prompts", type=str, default="eval/domain_prompts.json")
    parser.add_argument("--tag", type=str, required=True, help="e.g. 'base' or 'sft' -- labels rows for later comparison")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--output", type=str, default="eval/domain_eval/answers.jsonl")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = MODEL_PRESETS[args.preset]

    model = load_model_for_eval(args.checkpoint, args.preset, device)
    tokenizer = build_hf_tokenizer()
    prompts = json.loads(Path(args.prompts).read_text())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    with output_path.open("a") as f:
        for item in prompts:
            completion = greedy_generate(
                model, tokenizer, item["prompt"], preset.context_length,
                TokenizerConfig.eot_id, args.max_new_tokens,
            )
            row = {
                "tag": args.tag,
                "checkpoint": args.checkpoint,
                "prompt_id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "completion": completion,
                "timestamp": timestamp,
            }
            f.write(json.dumps(row) + "\n")
            print(f"[{item['id']}] {item['prompt']!r} -> {completion!r}")

    print(f"Appended {len(prompts)} rows to {output_path}")

if __name__ == "__main__":
    main()
