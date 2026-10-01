# TinyForge Notes

# Part 1: Data Pipeline

## Dataset

### Download

We used the `HuggingFaceFW/fineweb-edu` dataset, `sample-10BT` subset, with an approximate budget of 3B tokens.

A custom downloader fetched the data in chunks and stored it as parquet files. It took 9 minutes 19 seconds (5,345,689.5 tokens/s).

The data was split into training and validation sets:

| Split | Docs |
|---|---|
| Train | 2,905,409 |
| Val | 14,591 |

### Preprocessing

The dataset is the "edu" version, which already has some filtering and cleanup applied, but we ran additional filters on top.

**Gopher filter**

| Split | Kept | Dropped |
|---|---|---|
| Train | 2,896,126 | 9,283 |
| Val | 14,543 | 48 |

The quality filter (custom model) was skipped to save compute, since the source dataset is already high quality. A random sample was set aside for future ablation experiments.

**Deduplication**

We deduplicated in two passes:

| Method | Train | Val |
|---|---|---|
| Exact dedup | 2,861,770 | 14,542 |
| MinHash dedup | 2,846,154 | 14,542 |

### Results

Train count through the pipeline:

| Stage | Train docs | Change |
|---|---|---|
| Post-split | 2,905,409 | |
| Gopher filter | 2,896,126 | -9,283 |
| Exact dedup | 2,861,770 | -34,356 |
| MinHash near-dedup | 2,846,154 | -15,616 |

Val stayed at 14,542 through both dedup stages. Zero val docs were dropped, because the split-aware dedup rule always drops the train copy of a train/val duplicate and keeps val untouched. This is what prevents train/val leakage.

## Tokenizer

### Training

Trained on a 400MB random sample of the filtered and deduped train text (83,639 docs, seed 42).

| Metric | Value |
|---|---|
| Merges | 7,937 |
| Vocab size | 8,192 / 8,192 |
| Corpus preprocessing | 60.2s |
| BPE training | 147.5s |
| Total | 2m 27s |

Output saved to `tokenizer/vocab.json` and `tokenizer/merges.txt`.

### Encoding and sharding

Used HF's `encode_batch()` per chunk. This ran at roughly 650% CPU via native Rust thread parallelism (no explicit multiprocessing needed), at about 3,650 to 4,700 docs/s.

| Split | Tokens | Shards | Size |
|---|---|---|---|
| Train | 3,470,809,194 | 35 (34 full @ 100M + 1 partial @ 70,809,194) | 6.5GB |
| Val | 17,403,724 | 1 | 33MB |

Total elapsed: 13m 5s (13m 1s train, 3s val).

Note: EOT id is 256 (`special_tokens[0]`, right after the 256 raw bytes). This is not the "usually last vocab id" convention some guides use, noted here to avoid ambiguity in Part 2.

### Upload and verification

Uploaded to the private HF Hub dataset repo `Juanillaberia/TinyForge`: train shards, val shard, `meta.json`, `stats.json`. 41 files total, about 6.5GB.

Verified by downloading `vocab.json`, `merges.txt`, `meta.json`, and 3 shards (2 train, 1 val) fresh from the Hub into a scratch dir, building a tokenizer purely from those downloaded files, and decoding 5 random 400-token windows. All were real, coherent FineWeb-Edu text.

`meta.json` read back correctly: vocab_size 8192, train 3,470,809,194 tokens, val 17,403,724 tokens.

## Where We Are Now (Part 1)

Part 1 (data pipeline) is complete. The final dataset lives at [Juanillaberia/TinyForge](https://huggingface.co/datasets/Juanillaberia/TinyForge) on the HF Hub, verified end to end.

Part 2 (pretraining, on Colab GPU) starts from that Hub dataset and does not touch raw text again. Everything Part 2 needs:

- Tokenizer files: `vocab.json`, `merges.txt`
- Tokenized shards: 35 train shards, 1 val shard
- `meta.json` (vocab size, token counts) and `stats.json`
- EOT token id is 256, not the last vocab id

Open item carried over: the quality filter was not run, and a random sample was set aside for possible ablation experiments later.

# Part 2: Pretraining

Part 2 replaced the original single-phase plan with four staged parts, each ending in a usable
result: 2.a components, 2.b small-scale experiments, 2.c the real pretraining run, 2.d post-training.
Target architecture decided up front and not touched by the experiments below: 12 layers, d_model
640, 10 heads (head dim 64), SwiGLU d_ff 1,728, RMSNorm pre-norm, RoPE, context 1,024, vocab 8,192
with tied embeddings — ~59.5M non-embedding params, ~64.7M total.

## 2.a Components

Built the model, data loader and trainer shared by every later experiment and the final run:

- **Model** (`model/model.py`): RoPE, RMSNorm, SwiGLU, weight tying between the embedding and output
  head, GPT-2-style residual-projection init (std scaled by 1/√(2·n_layer)). Size is a config
  dataclass with named presets (`xs`, `s`, `m`, `base60m`), not hardcoded, so 2.b could sweep sizes
  from the same code.
- **Data loader** (`model/helper/data_loader.py`): memmap over the uint16 shards, random windows for
  train (seeded, resumable), fixed non-overlapping windows for val so val loss is comparable run to
  run.
- **Optimizers/schedules behind one flag each**: AdamW and Muon (`model/optim.py`), WSD and cosine
  (`model/scheduler.py`) — Muon applies only to the 7 in-block 2D hidden weight matrices per block;
  embeddings, head and norm gains always stay on AdamW.
- **Trainer** (`train/pretrain.py`): grad clip 1.0, gradient accumulation, W&B logging keyed on
  tokens seen (not steps), bf16/fp16 precision auto-detected per GPU, checkpoint/resume carrying
  model/optimizer/scaler/step/data-loader RNG state/W&B run id.

Gates passed before moving to 2.b: loss goes to ~0 on a single repeated batch for both optimizers; a
killed run resumes to the identical step, LR and loss; param counts for every preset match the
~12·d² per layer formula.

## 2.b Experiments

Four questions answered at small scale before committing compute to the real run (full tables, plots
and decisions in `EXPERIMENTS.md`):

| Experiment | Question | Decision |
|---|---|---|
| 0 — seed noise | How big is run-to-run noise at this scale? | ≈0.072-nat floor; <0.144 nats counts as a tie |
| A — optimizer | AdamW or Muon? | **Muon**, by 1.011 nats — decisively beyond noise |
| B — schedule | WSD or cosine? | **WSD** — cosine didn't clearly win |
| C — scaling | Is ~60M the right size? | Confirmed: loss keeps improving with size in this range, no change to the base60m plan |

Recipe locked into `configs/model.yaml`: Muon + WSD, `base60m`, 2.5B tokens.

## 2.c Pretraining — TinyForge-60M

**Full-size LR sweep**: three ~200-step calibration runs at `base60m`, starting at 2e-3/3e-3/4e-3
and extending to 6e-3/8e-3 once gains kept coming without a loss spike. Winner: **lr=8e-3** — the
last increment's gain (+0.072 nats) had decelerated into the Experiment 0 noise band.

**The long run**: Muon, WSD, lr=8e-3, 262,144 tokens/step (micro-batch 32 × grad-accum 8 × 1,024
context), 9,535 steps ≈ 2.5B tokens.

**Results**:

| Metric | Value |
|---|---|
| Final val loss | 2.7161 |
| Val perplexity | 15.12 |
| HellaSwag acc / acc_norm | 28.6% / 30.7% (random: 25%) |
| ARC-Easy acc / acc_norm | 46.8% / 41.5% |

Benchmarked with lm-evaluation-harness through a custom adapter (`model/evaluation.py`) wrapping
TinyForge's own tokenizer and model — both HellaSwag and ARC-Easy land modestly above their random
baselines, a real but limited signal given the token budget. Full report:
`eval/results/base60m_bench.json`.

**Domain eval**: a fixed 15-prompt set (`eval/domain_prompts.json`) run once before any SFT work,
saved to `eval/results/domain-answers.jsonl` (tag `base`). Fluent surface grammar, but frequent
greedy-decoding repetition loops and weak factual/arithmetic grounding — e.g. "Two plus two equals
one," and "The capital of France is the capital of the French Republic" (never names Paris). This is
the baseline the SFT model gets compared against in 2.d.

## Where We Are Now (Part 2)

2.a, 2.b and 2.c are done: components built and gated, recipe locked from small-scale experiments,
and the full `base60m` run finished with its benchmark numbers recorded. The last 2.c step is
pushing `TinyForge-60M` (checkpoint + model card) to the HF Hub. From there, 2.d (SFT, inference,
`TinyForge-60M-Instruct` release) is the remaining work — see `documentation/guide-remaining.md`.
