import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from config import TokenizerConfig, MODEL_PRESETS
from model.evaluation import load_model_for_eval
from model.generate import greedy_generate_ids
from tokenizer.chat_template import END_OF_TURN, encode_prompt
from tokenizer.tokenizer_hf import build_hf_tokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyForge domain-eval: run the fixed prompt set and save completions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Local path or '<hf_repo>/<...>/<file>.pt'")
    parser.add_argument("--preset", choices=list(MODEL_PRESETS), required=True)
    parser.add_argument("--prompts", type=str, default="eval/domain_prompts.json")
    parser.add_argument("--tag", type=str, required=True, help="e.g. 'base' or 'sft-50k' -- labels rows for later comparison")
    parser.add_argument("--chat", action="store_true",
                        help="Use each item's chat_prompt as a user turn in the chat template; stop at <|endofturn|>")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--output", type=str, default="eval/results/domain-answers.jsonl")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = MODEL_PRESETS[args.preset]

    model = load_model_for_eval(args.checkpoint, args.preset, device)
    tokenizer = build_hf_tokenizer()
    prompts = json.loads(Path(args.prompts).read_text())

    # Completion mode stops only at <|endoftext|>, as in the original base run; chat
    # mode also stops at the end of the assistant turn.
    stop_ids = {TokenizerConfig.eot_id}
    if args.chat:
        stop_ids.add(tokenizer.token_to_id(END_OF_TURN))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    with output_path.open("a") as f:
        for item in prompts:
            if args.chat:
                prompt = item["chat_prompt"]
                prompt_ids = encode_prompt([{"role": "user", "content": prompt}], tokenizer)
            else:
                prompt = item["prompt"]
                prompt_ids = tokenizer.encode(prompt).ids

            new_ids = greedy_generate_ids(model, prompt_ids, preset.context_length, stop_ids, args.max_new_tokens)
            stopped = bool(new_ids) and new_ids[-1] in stop_ids
            generated = tokenizer.decode(new_ids)
            # Completion rows keep the original format (prompt + continuation) so they
            # line up with the base rows recorded before --chat existed.
            completion = generated if args.chat else tokenizer.decode(prompt_ids + new_ids)

            row = {
                "tag": args.tag,
                "checkpoint": args.checkpoint,
                "prompt_id": item["id"],
                "category": item["category"],
                "mode": "chat" if args.chat else "completion",
                "prompt": prompt,
                "completion": completion,
                "num_new_tokens": len(new_ids),
                "stopped": stopped,
                "max_new_tokens": args.max_new_tokens,
                "timestamp": timestamp,
            }
            f.write(json.dumps(row) + "\n")
            print(f"[{item['id']}] {prompt!r} -> {completion!r}")

    print(f"Appended {len(prompts)} rows to {output_path}")

if __name__ == "__main__":
    main()
