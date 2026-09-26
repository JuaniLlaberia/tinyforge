# TinyForge — Part 1: Data + Tokenization Guide

Sep 25, 2026 · @Juan Ignacio Llaberia

## Purpose and scope

Part 1 ends with \~2.5B tokens of filtered, deduplicated FineWeb-Edu, encoded with your own 8,192-token BPE tokenizer and saved as training shards on the HF Hub or Drive. Part 2 (model, training, SFT, inference) starts from those shards and never touches raw text again.

Everything here runs on CPU: your laptop or a Colab CPU runtime. No GPU compute units are spent. Budget roughly 2–3 working days, most of it waiting on jobs.

This is the A4 pipeline applied to a real corpus, plus the A1 tokenizer trained on the output. The new parts are the scale, and a clean handoff format that Part 2 can rely on.

## Decisions to lock before starting

These are expensive to change once shards exist, so fix them on day one and write them into `data/config.yaml`.

| Decision | Value | Why |
| --- | --- | --- |
| Source | `HuggingFaceFW/fineweb-edu`, `sample-10BT` subset, streamed | Already educational-filtered; the 10B sample is more than you need |
| Token budget | \~2.5B tokens (own tokenizer) | 2–3× Chinchilla for \~60M params, per PROJECT\_DESIGN §5 |
| Vocab size | 8,192 incl. special tokens | Small tier; fits in uint16 |
| Special tokens | `<\|endoftext\|>` + 3 reserved chat tokens (user, assistant, end of turn) | Reserving them now means no vocab resize at SFT (Part 2, Phase 4) |
| Validation split | 0.5% of documents, by hash of doc ID | Deterministic, reproducible, split before dedup |
| Shard format | uint16 `.bin`, 100M tokens each | \~200 MB per shard, simple memmap loading |
| Storage | Private HF Hub dataset repo (Drive as backup) | Survives Colab resets, easy to pull in Part 2 |

One note on the budget: an 8k vocab splits text into more tokens than GPT-2's 50k vocab does. So 2.5B of your tokens is roughly 2B of the GPT-2 tokens FineWeb-Edu reports (approximate; you'll measure the real ratio in step 5). That means streaming about a quarter of `sample-10BT`, plus margin for what filtering removes.

## Step-by-step plan

Each step reads the previous step's output from disk and writes its own, so any step can be rerun alone. Log document counts and removal rates at every step; they become the data table in REPORT.md.

**Step 0 — Scaffold (30 min).** Create the repo layout from PROJECT\_DESIGN §8. Put the config from the table above in `data/config.yaml`. Make every script take `--input`, `--output`, and `--limit` so you can test on 10k docs first.

**Step 1 — Download (2–4h, mostly waiting).** Stream `sample-10BT` with `datasets` (`streaming=True`) and write JSONL or parquet chunks of \~100k docs each. Keep `id`, `text`, `url`, and `score`. Stop at \~30% of the subset; you can always pull more later.

**Step 2 — Hold out validation (15 min).** Hash each doc `id`; if `hash % 200 == 0`, it goes to `val/`. Do this now, before filtering or dedup touches anything.

**Optional step 2b — Keep an unfiltered sample (15 min).** Copy a random \~400M-token slice of the *train* docs (about 1.5–2 GB of text) to `raw_sample/` before steps 3–4 touch it, and never delete it. It lets you run a filtered-vs-unfiltered ablation in Part 2: two small models, same budget, one on this sample and one on an equal slice of the cleaned data. Tokenize it with the same tokenizer in step 6, into its own shards. Skip this step if you're sure you won't run the experiment; it can't be recreated later without re-downloading.

**Step 3 — Quality filters (2–3h).** Reuse your A4 filters: Gopher-style rules (word count, mean word length, symbol ratio, bullet/ellipsis lines), repetition ratios, and PII masking. Skip the language-ID and quality-classifier steps; FineWeb-Edu already applied both. Expect a low removal rate (a few percent). A high rate means a bug, not bad data.

**Step 4 — Deduplication (2–4h).** Exact dedup first (hash of normalized text), then MinHash near-dedup with your A4 code (or `text-dedup` if yours is too slow at this scale). FineWeb-Edu is deduplicated only within each crawl, so you will find cross-crawl duplicates. Run it over train and val together, and when a cluster spans both, drop the train copies. That is what prevents leakage.

**Step 5 — Train the tokenizer (1–3h).** Train your A1 BPE on a 300–500 MB random sample of the filtered, deduped train text, vocab 8,192. Then run the sanity checks: numbers, whitespace runs, code, accented and non-Latin text, URLs. Record bytes-per-token on val; that number goes in the report.

**Step 6 — Encode and shard (2–4h).** A pure-Python A1 encoder is far too slow for billions of tokens. Load your trained merges into a fast encoder (tiktoken with custom ranks, or HF `tokenizers`) and confirm it produces identical IDs to your A1 encoder on 10k docs. Then encode with multiprocessing, append `<|endoftext|>` after each doc, and write 100M-token uint16 shards. Val gets its own shard(s).

**Step 7 — Upload and verify (1h).** Push shards, tokenizer files, and metadata to the HF Hub. Then, from a fresh Colab session, download one shard, decode 5 random windows, and read them. If they look like real text, Part 1 is done.

## Output contract (what Part 2 expects)

Part 2 assumes exactly this layout in the HF Hub repo. If something changes, bump the version in `meta.json` rather than editing in place.

| Path | Contents |
| --- | --- |
| `tokenizer/vocab.json`, `tokenizer/merges.txt` | Your trained BPE, loadable by both your A1 code and the fast encoder |
| `train/shard_0000.bin` … | uint16 token IDs, 100M per shard, docs separated by `<\|endoftext\|>` |
| `val/shard_0000.bin` | Same format, held-out docs only |
| `meta.json` | Version, vocab size, EOT id, token count per shard, bytes-per-token, git commit of the pipeline |
| `stats.json` | Doc counts after each step and removal rate per filter |

The raw and intermediate JSONL can stay local or on Drive. Only the files above are needed for training (about 5 GB for 2.5B tokens).

## Checklist and gotchas

- [ ] Config file written; every script tested on `--limit 10000` first
- [ ] Validation split made before any filtering or dedup
- [ ] Filter removal rates logged and plausible (a few percent)
- [ ] Dedup run across train + val; train copies of shared clusters dropped
- [ ] Tokenizer sanity-checked on numbers, whitespace, code, non-Latin text
- [ ] Fast encoder matches A1 encoder exactly on 10k docs
- [ ] Shards, tokenizer, `meta.json`, `stats.json` on the HF Hub
- [ ] Round-trip decode checked from a fresh session

Gotchas:

- **uint16 overflow.** Assert `max(ids) < 65536` before writing. It holds for 8,192 but guards against a config mistake later.
- **EOT id.** Pick it once (usually the last vocab id) and store it in `meta.json`. A mismatch between tokenizer and trainer silently breaks document boundaries.
- **Disk space.** Raw text for \~2.5B tokens is roughly 10 GB, and intermediates multiply that. Delete step outputs once the next step is verified, except the optional raw sample from step 2b.
- **Shuffling.** FineWeb-Edu arrives grouped by crawl. Shuffle documents before sharding so each shard is a representative mix.
- **Don't over-filter.** This corpus is already clean. The value of steps 3–4 is proving your pipeline works and measuring what it removes, not squeezing the data.
