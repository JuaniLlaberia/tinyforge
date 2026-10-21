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
and decisions in `train/results/EXPERIMENTS.md`):

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

## 2.d Post-training — SFT

Goal: measure what supervised fine-tuning changes in a 60M model — format, stopping, repetition,
staying on topic — not build a usable assistant. SFT teaches behavior; it can't add knowledge the
base model doesn't have.

### Chat format

`<|endoftext|><|user|>{user}<|endofturn|><|assistant|>{assistant}<|endofturn|>…`, using the three
chat tokens reserved in Part 1 (ids 258, 259, 257), each encoding to a single id. One module
(`tokenizer/chat_template.py`) builds both the training sequences and the inference prompts, and
the inference prompt is checked to be a strict prefix of the training sequence, so the two formats
can't drift apart. Loss is computed only on assistant replies plus their closing `<|endofturn|>`,
so the model learns when to stop. System messages are merged into the first user turn.

The chat tokens never appeared in pretraining, and because the embedding is tied to the output
head, pretraining only ever pushed their logits down. Their rows were reset to the mean embedding
before SFT.

### Data

`HuggingFaceTB/smol-smoltalk` (built by the SmolLM2 team for small models), plus 31 hand-written
research-style conversations. Conversations over the 1,024-token context are truncated at a turn
boundary — the longest prefix of complete user/assistant pairs that fits — instead of dropped: a
smoke run showed plain dropping lost 49% of conversations and cut 3+-turn chats from 59% to 26%,
because the largest source (`smol-magpie-ultra-short`, 59%) is mostly long 3-turn chats. With
truncation only 2.7% was dropped and the source mix stayed within ~1.5 points of the full dataset.

| Split | Conversations | Tokens | Turns 1 / 2 / 3+ |
|---|---|---|---|
| `sft-50k` train | 50,031 | 31.2M | 47% / 38% / 15% |
| `sft-2k` train (nested subset) | 2,031 | 1.24M | 49% / 38% / 13% |
| Held-out (dataset's `test` split) | 1,000 | 0.61M | 51% / 36% / 13% |

### Runs

Same recipe for both runs: start from the final base checkpoint, Muon + AdamW at peak LR 8e-4 (10x
below pretraining), 3% warmup then linear decay to 0, 64 conversations per step, 3 epochs. On an
A100 the 50k run took ~17 minutes at ~102k tokens/sec; all of 2.d's training used ~3 CU.

The two runs test LIMA's claim that ~1k–2k high-quality examples are enough for alignment, at a
model ~1,000x smaller than LIMA's.

| | Base | `sft-2k` | `sft-50k` |
|---|---|---|---|
| Held-out SFT loss | 2.474 | 1.834 | **1.375** |
| Optimizer steps | — | 96 | 2,346 |

`sft-50k` wins by 0.46 nats, over 6x the seed-noise floor from 2.b. Per update the two runs
improved about equally (`sft-2k` after 96 steps: 1.834; `sft-50k` at step 100: 1.775), so
repeating 2k examples three times didn't hurt — the small run simply got ~24x fewer updates.
`sft-50k` had plateaued by the end (epoch 3 gained 0.002 nats over epoch 2).

### Behavior: domain eval

The 15 domain-eval prompts from 2.c, each rephrased as a question (`chat_prompt`) for the SFT
models. Greedy decoding; base answers capped at 60 tokens (as recorded in 2.c), SFT answers at
256. Full answers: `eval/results/domain-comparison.md`.

| | Base | `sft-2k` | `sft-50k` |
|---|---|---|---|
| Stops on its own | 0/15 | 0/15 | **5/15** |
| Repeated 4-grams, first 60 tokens | 0.63 | 0.28 | **0.09** |
| On-topic (manual) | 10/15 | 11/15 | 11/15 |
| Correct (12 factual/reasoning/code prompts) | 1/12 | 1/12 | **3/12** |

- `sft-50k` is the only model that ends its turn, and its answers are mostly loop-free: "The
  capital of France is Paris…", "…is Jupiter…", a correct `reverse_string` returning `s[::-1]`,
  a fluent story opening.
- `sft-2k` learned the openings ("Here's how you can implement…") but never stops, and over 256
  tokens loops worse than the base model. The behavioral result agrees with held-out loss: 2k
  examples aren't enough at 60M.
- Base names neither Paris nor Jupiter in completion mode; `sft-50k` names both. At least some
  facts were already in the base weights, and SFT made them come out. 15 prompts is too few to
  claim more than that.
- Still wrong after SFT: arithmetic ("two plus two" loops), the train-time question, the syllogism
  (it treats Whiskers as a cat species), Fibonacci and `is_even` code, and "transformer" (always
  answered as the electrical device — FineWeb-Edu has far more of that sense).

### Benchmarks

Same lm-evaluation-harness setup as 2.c (0-shot, full tasks, scored without the chat template).

| | Base | `sft-50k` | Difference |
|---|---|---|---|
| HellaSwag acc / acc_norm | 28.6% / 30.7% | 28.7% / 30.9% | +0.1 / +0.2 pts (noise) |
| ARC-Easy acc / acc_norm | 46.8% / 41.5% | 43.4% / 39.4% | -3.4 / -2.0 pts |

HellaSwag is unchanged. ARC-Easy acc drops 3.4 points (2.4 standard errors), a small cost of SFT;
acc_norm's 2.0-point drop is within noise. These tasks score raw text continuations, which is the
base model's format and no longer the SFT model's, so part of the drop may be format mismatch
rather than lost knowledge.

**Decision:** `sft-50k` epoch 3 ships as `TinyForge-60M-Instruct`.

## Where We Are Now (Part 2)

2.a–2.c are done and `TinyForge-60M` is public on the HF Hub. In 2.d, SFT is done: `sft-50k`
epoch 3 is chosen for `TinyForge-60M-Instruct` (checkpoints are in a private HF repo). What's left:
the weights-only upload and model card, then inference (KV cache, sampling, CLI chat) and
`REPORT.md` — see `documentation/guide-remaining.md`.
