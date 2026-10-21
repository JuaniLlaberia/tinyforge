import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from huggingface_hub import upload_folder
from tqdm import tqdm

from config import GlobalConfig, SFTConfig, TokenizerConfig, MODEL_PRESETS
from data.progress import format_duration
from tokenizer.chat_template import encode_conversation, normalize_messages, render
from tokenizer.tokenizer_hf import build_hf_tokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyForge SFT data: smol-smoltalk -> filtered, sampled JSONL splits")
    parser.add_argument("--num-train", type=int, default=SFTConfig.num_train)
    parser.add_argument("--num-ablation", type=int, default=SFTConfig.num_ablation)
    parser.add_argument("--num-heldout", type=int, default=SFTConfig.num_heldout)
    parser.add_argument("--seed", type=int, default=GlobalConfig.seed)
    parser.add_argument("--output", type=str, default=str(SFTConfig.output_path))
    parser.add_argument("--hf-repo-id", type=str, default=None,
                        help="Dataset repo to push the output folder to (e.g. Juanillaberia/TinyForge); skipped if not given")
    return parser.parse_args()

def conversation_hash(messages: list[dict]) -> str:
    return hashlib.sha1(render(messages).encode("utf-8")).hexdigest()

def turns_bucket(num_turns: int) -> str:
    return str(num_turns) if num_turns < 3 else "3+"

def encode_example(raw_messages: list[dict], source: str, tokenizer, max_tokens: int) -> tuple[dict | None, str, int]:
    """
    Normalizes and tokenizes one conversation and decides whether it's kept.
    Conversations over max_tokens are truncated at a turn boundary: the longest
    prefix of complete user/assistant pairs that fits is kept, so no assistant
    reply is ever cut mid-way. Only conversations whose first pair alone doesn't
    fit are dropped.

    Args:
        raw_messages (list[dict]): smol-smoltalk-style messages.
        source (str): Source label, carried into the output row.
        tokenizer: HF tokenizer from build_hf_tokenizer().
        max_tokens (int): Max sequence length in tokens (context_length + 1, since
            training feeds ids[:-1] as inputs and ids[1:] as targets).
    Returns:
        tuple[dict | None, str, int]: (example, reason, original_turns). reason is
            "kept", "truncated", "too_long", "special_token" or "malformed";
            example is None unless reason is "kept" or "truncated".
            original_turns is the number of complete pairs before truncation (0
            when the conversation couldn't be normalized).
    """
    messages = normalize_messages(raw_messages)
    if messages is None:
        has_special = any(token in m["content"] for m in raw_messages for token in TokenizerConfig.special_tokens)
        return None, "special_token" if has_special else "malformed", 0

    # A trailing user turn has no reply to train on; complete pairs only.
    original_turns = len(messages) // 2
    if original_turns == 0:
        return None, "malformed", 0

    for turns in range(original_turns, 0, -1):
        ids, train_mask = encode_conversation(messages[:2 * turns], tokenizer)
        if len(ids) <= max_tokens:
            example = {
                "messages": messages[:2 * turns],
                "source": source,
                "num_tokens": len(ids),
                "num_trained_tokens": sum(train_mask),
                "num_turns": turns,
            }
            return example, "kept" if turns == original_turns else "truncated", original_turns

    return None, "too_long", original_turns

def collect(rows: Iterable[dict], target: int, tokenizer, max_tokens: int, exclude: set[str], desc: str) -> tuple[list[dict], set[str], dict]:
    """
    Walks rows in (already shuffled) order until `target` conversations are kept.
    Because the order is random, the drop counts are an unbiased sample of the
    whole split.

    Args:
        rows (Iterable[dict]): Rows with "messages" and "source".
        target (int): Number of conversations to keep.
        tokenizer: HF tokenizer from build_hf_tokenizer().
        max_tokens (int): See encode_example.
        exclude (set[str]): Conversation hashes that must not be kept (e.g. train,
            when collecting held-out).
        desc (str): Progress bar label.
    Returns:
        tuple[list[dict], set[str], dict]: (kept examples, their hashes, pool stats).
    """
    kept, kept_hashes = [], set()
    reasons = Counter()
    turns_valid, turns_kept = Counter(), Counter()
    valid_by_source, truncated_by_source, too_long_by_source = Counter(), Counter(), Counter()

    with tqdm(total=target, desc=desc) as bar:
        for row in rows:
            example, reason, original_turns = encode_example(row["messages"], row["source"], tokenizer, max_tokens)
            if original_turns > 0:
                turns_valid[turns_bucket(original_turns)] += 1
                valid_by_source[row["source"]] += 1
            if reason == "truncated":
                truncated_by_source[row["source"]] += 1
            if reason == "too_long":
                too_long_by_source[row["source"]] += 1
            if example is not None:
                h = conversation_hash(example["messages"])
                if h in exclude or h in kept_hashes:
                    reason = "duplicate"
                else:
                    kept.append(example)
                    kept_hashes.add(h)
                    turns_kept[turns_bucket(example["num_turns"])] += 1
                    bar.update(1)
            reasons[reason] += 1
            if len(kept) == target:
                break

    if len(kept) < target:
        raise RuntimeError(f"{desc}: only {len(kept)}/{target} conversations survived filtering")

    examined = sum(reasons.values())
    stats = {
        "examined": examined,
        "outcomes": dict(reasons),
        "drop_rate": round(1 - (reasons["kept"] + reasons["truncated"]) / examined, 4),
        # Turn mix before vs after turn-boundary truncation: shows how much the
        # 1,024-token limit shifts the sample toward fewer turns.
        "turns_valid": dict(sorted(turns_valid.items())),
        "turns_kept": dict(sorted(turns_kept.items())),
        "truncated_rate_by_source": {s: round(truncated_by_source[s] / n, 4) for s, n in valid_by_source.most_common()},
        "too_long_rate_by_source": {s: round(too_long_by_source[s] / n, 4) for s, n in valid_by_source.most_common()},
    }
    return kept, kept_hashes, stats

def summarize(examples: list[dict]) -> dict:
    """
    Token, turn and source stats for one output split.
    """
    tokens = [e["num_tokens"] for e in examples]
    trained = sum(e["num_trained_tokens"] for e in examples)
    sources = Counter(e["source"] for e in examples)
    return {
        "conversations": len(examples),
        "tokens_total": sum(tokens),
        "tokens_mean": round(statistics.mean(tokens), 1),
        "tokens_median": statistics.median(tokens),
        "tokens_p95": round(statistics.quantiles(tokens, n=20)[18]) if len(tokens) > 1 else tokens[0],
        "tokens_max": max(tokens),
        "trained_tokens_total": trained,
        "trained_fraction": round(trained / sum(tokens), 4),
        "turns": dict(sorted(Counter(turns_bucket(e["num_turns"]) for e in examples).items())),
        "source_mix": {s: round(n / len(examples), 4) for s, n in sources.most_common()},
    }

def load_research_prompts(path: Path, tokenizer, max_tokens: int) -> list[dict]:
    """
    Loads the hand-written research conversations. Unlike smol-smoltalk rows,
    every one of these must pass filtering -- a drop here is a bug in the file.
    """
    examples = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        example, reason, _ = encode_example(json.loads(line)["messages"], SFTConfig.research_source, tokenizer, max_tokens)
        if reason != "kept":
            raise ValueError(f"{path}:{line_no} rejected ({reason})")
        examples.append(example)
    return examples

def write_jsonl(path: Path, examples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

def main() -> None:
    args = parse_args()
    start_time = time.monotonic()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = build_hf_tokenizer()
    max_tokens = MODEL_PRESETS[SFTConfig.preset].context_length + 1

    research = load_research_prompts(SFTConfig.research_prompts_path, tokenizer, max_tokens)
    research_hashes = {conversation_hash(e["messages"]) for e in research}
    print(f"Loaded {len(research)} research conversations")

    train_split = load_dataset(SFTConfig.dataset, split="train")
    test_split = load_dataset(SFTConfig.dataset, split="test")
    full_train_sources = Counter(train_split["source"])

    # Shuffled walk; the ablation set is the first num_ablation kept conversations,
    # so it's a random subset nested inside the main set.
    sampled, train_hashes, train_pool_stats = collect(
        train_split.shuffle(seed=args.seed), args.num_train, tokenizer, max_tokens,
        exclude=research_hashes, desc="train",
    )
    heldout, _, heldout_pool_stats = collect(
        test_split.shuffle(seed=args.seed), args.num_heldout, tokenizer, max_tokens,
        exclude=train_hashes | research_hashes, desc="heldout",
    )

    train_main = sampled + research
    train_ablation = sampled[:args.num_ablation] + research

    write_jsonl(output_path / SFTConfig.train_main_file, train_main)
    write_jsonl(output_path / SFTConfig.train_ablation_file, train_ablation)
    write_jsonl(output_path / SFTConfig.heldout_file, heldout)

    stats = {
        "dataset": SFTConfig.dataset,
        "seed": args.seed,
        "max_tokens": max_tokens,
        "full_train_source_mix": {s: round(n / len(train_split), 4) for s, n in full_train_sources.most_common()},
        "train_pool": train_pool_stats,
        "heldout_pool": heldout_pool_stats,
        "splits": {
            SFTConfig.train_main_file: summarize(train_main),
            SFTConfig.train_ablation_file: summarize(train_ablation),
            SFTConfig.heldout_file: summarize(heldout),
            "research_prompts": summarize(research),
        },
    }
    (output_path / SFTConfig.stats_file).write_text(json.dumps(stats, indent=2))

    print(json.dumps(stats, indent=2))
    print(f"Wrote {output_path} in {format_duration(time.monotonic() - start_time)}")

    if args.hf_repo_id:
        upload_folder(repo_id=args.hf_repo_id, repo_type="dataset", folder_path=str(output_path),
                      path_in_repo=SFTConfig.hub_path_in_repo)
        print(f"Pushed {output_path} to {args.hf_repo_id}/{SFTConfig.hub_path_in_repo}")

if __name__ == "__main__":
    main()
