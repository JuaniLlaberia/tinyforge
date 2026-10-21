"""
Base-vs-SFT domain-eval comparison (documentation/guide-remaining.md, SFT Step 5).

Reads eval/results/domain-answers.jsonl and writes a markdown report: a summary
table of the automatic metrics per tag, an on-topic table to fill in by hand, and
every prompt's answers side by side. Metrics are computed on the generated tokens
only (the prompt is stripped from completion-mode rows).
"""
import argparse
import json
from pathlib import Path

from tokenizer.tokenizer_hf import build_hf_tokenizer

LEGACY_MAX_NEW_TOKENS = 60  # base rows were written before rows recorded their limit
# Repetition is also measured on this many leading tokens for every tag: base answers
# were capped at 60 and SFT answers at 256, and longer answers have more room to loop.
COMPARABLE_PREFIX_TOKENS = 60

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyForge base-vs-SFT domain-eval comparison")
    parser.add_argument("--answers", type=str, default="eval/results/domain-answers.jsonl")
    parser.add_argument("--prompts", type=str, default="eval/domain_prompts.json")
    parser.add_argument("--judgments", type=str, default="eval/results/domain-judgments.json",
                        help="Manual on-topic/correct labels; kept out of the report so reruns don't erase them")
    parser.add_argument("--tags", type=str, default="base,sft-2k,sft-50k", help="Comma-separated, in column order")
    parser.add_argument("--max-chars", type=int, default=400, help="Truncate answers in the report")
    parser.add_argument("--output", type=str, default="eval/results/domain-comparison.md")
    return parser.parse_args()

def latest_rows(path: Path, tags: list[str]) -> dict[str, dict[str, dict]]:
    """
    tag -> prompt_id -> row; a later row for the same (tag, prompt_id) replaces an
    earlier one, so rerunning a tag doesn't need the old rows deleted.
    """
    rows = {tag: {} for tag in tags}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["tag"] in rows:
            rows[row["tag"]][row["prompt_id"]] = row
    return rows

def generated_text(row: dict) -> str:
    if row.get("mode", "completion") == "chat":
        return row["completion"]
    return row["completion"][len(row["prompt"]):]

def repeated_ngram_rate(ids: list[int], n: int = 4) -> float:
    """
    Fraction of n-grams that repeat an earlier n-gram: 0 = no repetition, near 1 =
    a loop.
    """
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0

def row_metrics(row: dict, tokenizer) -> dict:
    # Re-encoding decoded text can differ from the generated ids by a token, so the
    # recorded count is preferred; legacy rows only have the text.
    ids = tokenizer.encode(generated_text(row)).ids
    tokens = row.get("num_new_tokens", len(ids))
    max_new_tokens = row.get("max_new_tokens", LEGACY_MAX_NEW_TOKENS)
    # Legacy rows don't record whether generation stopped; with no stop token kept in
    # the text, an answer shorter than the limit can only have ended on a stop token.
    stopped = row["stopped"] if "stopped" in row else tokens < max_new_tokens
    return {"tokens": tokens, "stopped": stopped, "repetition": repeated_ngram_rate(ids),
            "repetition_prefix": repeated_ngram_rate(ids[:COMPARABLE_PREFIX_TOKENS]),
            "max_new_tokens": max_new_tokens, "mode": row.get("mode", "completion")}

def main() -> None:
    args = parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    prompts = json.loads(Path(args.prompts).read_text())
    rows = latest_rows(Path(args.answers), tags)
    tokenizer = build_hf_tokenizer()

    missing = {tag: [p["id"] for p in prompts if p["id"] not in rows[tag]] for tag in tags}
    if any(missing.values()):
        raise ValueError(f"Missing answers: { {t: ids for t, ids in missing.items() if ids} }")

    metrics = {tag: {p["id"]: row_metrics(rows[tag][p["id"]], tokenizer) for p in prompts} for tag in tags}

    lines = ["# Domain eval: base vs SFT", "",
             "Greedy decoding. Base answers continue the completion prompt; SFT answers reply to the",
             "`chat_prompt` through the chat template. Metrics use generated tokens only.", "",
             f"| Tag | Mode | Max new tokens | Stop rate | Repeated-4-gram rate, first {COMPARABLE_PREFIX_TOKENS} tokens "
             "| Repeated-4-gram rate, whole answer | Mean answer tokens |",
             "| --- | ---- | -------------- | --------- | --- | --- | ------------------ |"]
    for tag in tags:
        m = list(metrics[tag].values())
        lines.append(f"| `{tag}` | {m[0]['mode']} | {m[0]['max_new_tokens']} "
                     f"| {sum(x['stopped'] for x in m)}/{len(m)} "
                     f"| {sum(x['repetition_prefix'] for x in m) / len(m):.2f} "
                     f"| {sum(x['repetition'] for x in m) / len(m):.2f} "
                     f"| {sum(x['tokens'] for x in m) / len(m):.0f} |")

    judgments_path = Path(args.judgments)
    judgments = json.loads(judgments_path.read_text()) if judgments_path.exists() else {}
    criteria = judgments.get("_criteria", {})
    for field, title in (("on_topic", "On-topic"), ("correct", "Correct")):
        lines += ["", f"## {title} (manual)", "", f"_{criteria.get(field, '')}_", "",
                  "| Prompt | " + " | ".join(f"`{t}`" for t in tags) + " |",
                  "| ------ | " + " | ".join("---" for _ in tags) + " |"]
        labels = {tag: [judgments.get(tag, {}).get(p["id"], {}).get(field, "?") for p in prompts] for tag in tags}
        lines += [f"| {p['id']} | " + " | ".join(labels[tag][i] for tag in tags) + " |" for i, p in enumerate(prompts)]
        totals = [f"{labels[tag].count('y')}/{sum(label in ('y', 'n') for label in labels[tag])}" for tag in tags]
        lines.append("| **Total y** | " + " | ".join(totals) + " |")
    if criteria.get("status"):
        lines += ["", f"Judgments: {criteria['status']} (`{judgments_path}`)."]
    lines += ["", "## Answers", ""]

    for p in prompts:
        lines += [f"### {p['id']} ({p['category']})", "",
                  f"Completion prompt: `{p['prompt']!r}` · Chat prompt: \"{p['chat_prompt']}\"", ""]
        for tag in tags:
            m = metrics[tag][p["id"]]
            text = generated_text(rows[tag][p["id"]]).strip()
            if len(text) > args.max_chars:
                text = text[:args.max_chars] + " [...]"
            lines += [f"**`{tag}`** ({m['tokens']} tokens, {'stopped' if m['stopped'] else 'hit limit'}, "
                      f"repetition {m['repetition']:.2f})", "", "````text", text, "````", ""]

    output_path = Path(args.output)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:7 + len(tags)]))
    print(f"\nWrote {output_path}")

if __name__ == "__main__":
    main()
