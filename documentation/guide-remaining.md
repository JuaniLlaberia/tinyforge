# TinyForge — what's left

Oct 1, 2026 · @Juan Ignacio Llaberia · SFT plan added Oct 21, 2026

Replaces `guide-part2.md` and `guide-part2-revised.md`, both now executed through 2.c. This guide
covers only what's still open: finishing 2.c's release, and all of 2.d.

Full history of how the recipe was chosen lives in `train/results/EXPERIMENTS.md`; the Part 2 pretraining results
(val perplexity 15.12, HellaSwag/ARC-Easy, domain-eval baseline) are written up in
`documentation/writeup.md`.

## Finish 2.c — release TinyForge-60M

`Juanillaberia/TinyForge-60M` is public with its model card and final checkpoint. Remaining fixes
to the HF `README.md`:

- [ ] Replace both `<GITHUB_REPO_URL>` placeholders with `https://github.com/JuaniLlaberia/tinyforge`.
- [ ] Limitations: 2.5B tokens is ~2x *above* compute-optimal for this size, not "far below"; reword.
- [ ] Peak LR row: make the sweep description match `EXPERIMENTS.md` (4e-3/6e-3/8e-3 table).
- [ ] Commit the card as `documentation/model_card_TinyForge-60M.md` (README links to it), and
      update `configs/model.yaml` with the final lr 8e-3.
- [ ] Confirm the model loads cleanly from a fresh clone using the model card's own usage snippet.

## 2.d Post-training and release → TinyForge-60M-Instruct

**Goal**: measure what SFT changes at 60M (format, stopping, repetition, staying on topic), not
build a usable assistant. Correctness is bounded by what the base model knows, and SFT doesn't add
knowledge.

**Compute**: 30.55 CU left as of Oct 21. At ~5.17 CU per A100-hour, that's ~6 A100-hours. 2.d
plans ~5 CU; the rest is reserve and stretch goals. All code is written and smoke-tested on CPU
locally; A100 sessions are only for real runs, disconnected as soon as the checkpoint is pushed.

### SFT — locked decisions

| Decision | Value |
|---|---|
| Source | `HuggingFaceTB/smol-smoltalk` (460,341 train / 24,229 test conversations, 10 sources) |
| Main run (`sft-50k`) | 50,000 conversations, random sample (seed 42) of the length-filtered train split, plus the hand-written research prompts |
| Ablation (`sft-2k`) | Nested random 2,000-conversation subset of the 50k, plus the same research prompts. Tests LIMA's small-data claim at 60M |
| Held-out | 1,000 conversations from the `test` split, same filtering; shared by both runs |
| Chat format | `<\|endoftext\|><\|user\|>{u}<\|endofturn\|><\|assistant\|>{a}<\|endofturn\|>…` |
| Loss mask | Assistant content plus its closing `<\|endofturn\|>` only; everything else `-100` |
| System messages | Merged into the first user turn as `{system}\n\n{user}` |
| Long conversations | Over 1,024 tokens: truncated at a turn boundary (longest prefix of complete user/assistant pairs that fits); dropped only if the first pair alone doesn't fit. Plain dropping cut 3+-turn conversations from 59% to 26% of the sample in a 585-conversation smoke run |
| Batching | Right-padded, length-bucketed; no packing. Input pad id 256, label pad -100 |
| Chat-token init | Rows 257–259 of the tied embedding set to the mean of the non-special rows before step 0 |
| Init weights | `Juanillaberia/TinyForge-60M/base60m-pretrain/step_0009535.pt`, weights only; fresh optimizers |
| Optimizer | Muon (hidden matrices) + AdamW (embeddings/norms), both peak 8e-4, weight decay 0.1 |
| Schedule | 3% warmup, then linear decay to 0 (`build_scheduler("wsd", warmup_frac=0.03, decay_frac=0.97)`) |
| Batch | 64 conversations/step (micro-batch 32 × grad-accum 2) |
| Epochs | 3 max; held-out loss after every epoch; keep each run's best epoch |
| Loss normalization | Sum of token losses over the whole step ÷ assistant tokens in the step (not a mean of per-micro-batch means) |
| Precision | bf16 (A100) |

At batch 64, `sft-2k` gets ~95 optimizer steps vs ~2,350 for `sft-50k`. That's the point of the
ablation (same recipe, less data); say so in the report.

### Files

| File | | Purpose |
|---|---|---|
| `config.py` | changed | `SFTConfig`: every value in the table above, plus paths |
| `tokenizer/chat_template.py` | new | Single source of truth for the format: `normalize_messages`, `encode_conversation` → (ids, train_mask), `encode_prompt` → ids ending in `<\|assistant\|>`. Used by SFT, domain eval and the CLI chat |
| `data/sft/research_prompts.jsonl` | new | ~30 hand-written research-style conversations, same `messages` schema as smol-smoltalk. Must not overlap the domain-eval prompts |
| `train/prepare_sft_data.py` | new | CPU: download → normalize → tokenize → length filter → sample 50k / nested 2k / held-out 1k → JSONL + stats → push to `Juanillaberia/TinyForge` under `sft/` |
| `model/helper/sft_loader.py` | new | Loads the JSONL, tokenizes through `chat_template`, seeded per-epoch shuffle, length buckets, padding |
| `train/sft.py` | new | Trainer: load base weights, chat-token init, masked loss, held-out eval, per-epoch checkpoints to `Juanillaberia/TinyForge-60M-Instruct/<run_name>/`, W&B, fixed chat samples |
| `model/generate.py` | changed | `greedy_generate` takes prompt ids and a set of stop ids (259, 256); can return only the new tokens. The KV cache extends this same function later |
| `model/domain_eval.py` | changed | `--chat`: uses `chat_prompt` + `encode_prompt`, stops at `<\|endofturn\|>`, `--max-new-tokens 256` |
| `eval/domain_prompts.json` | changed | `chat_prompt` added to each of the 15 items (same ids/categories) |
| `eval/compare_domain.py` | new | Base-vs-SFT table and metrics from `domain-answers.jsonl` |
| `train/push_to_hub.py` | new | Strips a training checkpoint to weights only and uploads it, with its model card, to the root of an HF model repo |
| `documentation/model_card_TinyForge-60M-Instruct.md` | new | Instruct model card, pushed as the repo's `README.md` |

Reused unchanged: `build_optimizers`, `build_scheduler`, `save_checkpoint`, `detect_precision`,
`resolve_checkpoint_source`. SFT checkpoints keep the `model_state_dict` key, so
`load_model_for_eval` (and through it `evaluation.py` and `domain_eval.py`) loads them as-is.

### Steps and gates

**Colab workflow**: same as 2.c. Every step is a `python -m ...` CLI run as a Colab cell, and every
step's inputs and outputs live on the HF Hub, so any fresh session can pick up where the last one
stopped:

| Artifact | Hub location |
|---|---|
| Base weights (input) | `Juanillaberia/TinyForge-60M/base60m-pretrain/step_0009535.pt` |
| SFT data (Step 2) | `Juanillaberia/TinyForge` dataset repo, `sft/` |
| Per-epoch SFT checkpoints (Step 4) | `Juanillaberia/TinyForge-60M-Instruct/<run_name>/` (private) |
| Released weights + card (Step 6) | `Juanillaberia/TinyForge-60M-Instruct` repo root |

**Step 1 — Chat template** (local, 0 CU)

- [x] `tokenizer/chat_template.py`
- Gates:
  - Each special token encodes to a single id; `decode(ids, skip_special_tokens=False)` reproduces
    the template string exactly.
  - Decoding only the masked-in target positions gives exactly the assistant contents, each followed
    by `<|endofturn|>`. Check on 3 real conversations: single-turn, multi-turn, and one with a
    system message.
  - `encode_prompt(msgs)` is a strict prefix of `encode_conversation(msgs + [assistant_reply])`, so
    the training and inference formats can't drift apart.

**Step 2 — Data** (local, 0 CU)

- [x] Write `data/sft/research_prompts.jsonl`.
- [x] `train/prepare_sft_data.py`, then push the outputs to the HF dataset repo.
- Gates:
  - Stats logged and recorded in `EXPERIMENTS.md`: conversations after normalization, dropped
    (malformed / too long), tokens per conversation (mean, median, p95), source mix of the 50k vs
    the full train split, fraction of tokens that are assistant (trained) tokens.
  - No held-out conversation also appears in train (hash of normalized text).
  - 5 random encoded examples decoded and read by hand.

**Step 3 — Loader and trainer** (local CPU, 0 CU)

- [x] `model/helper/sft_loader.py`, `train/sft.py`, `train/sft_gates.py` (static gates: `python -m train.sft_gates`)
- Gates:
  - `xs` preset, random init, one repeated batch: masked loss goes to ~0.
  - Normalization check: a batch where every label is `-100` except one token gives exactly that
    token's cross-entropy.
  - After chat-token init, rows 257–259 equal the mean row. Held-out loss of the base model at
    step 0 is logged (the "before" number).
  - 20 steps of `base60m` on the real 2k file run end to end: held-out eval, checkpoint written,
    checkpoint reloads through `load_model_for_eval`. CPU training ran on ~1 core and was too slow
    for this, so the A100 smoke run (Step 4) covered it instead: base held-out loss 2.47 → 1.92
    after 50 steps, ~101k tokens/sec, checkpoint reloaded with the head still tied.

**Step 4 — Runs** (A100, ~3.5 CU)

Colab cells (the Step 1–3 code must be committed and pushed first, so the clone has it):

```bash
!nvidia-smi
!git clone -b dev https://github.com/JuaniLlaberia/tinyforge.git
%cd tinyforge
!pip install -q -r requirements.txt
```

```python
from huggingface_hub import login; login()   # HF token with write access
import wandb; wandb.login()
```

```bash
# Smoke: ~50 steps, nothing pushed. Check tokens/sec, peak memory, and the samples.
!python -m train.sft --run-name sft-50k --max-steps 50

# Real runs; each pushes epoch_{1,2,3}.pt to the private Instruct repo
!python -m train.sft --run-name sft-50k --hf-repo-id Juanillaberia/TinyForge-60M-Instruct
!python -m train.sft --run-name sft-2k  --hf-repo-id Juanillaberia/TinyForge-60M-Instruct

# If a session dies mid-run: resume from the last finished epoch (same flags plus this)
# --resume-from Juanillaberia/TinyForge-60M-Instruct/sft-50k/epoch_1.pt
```

- [x] Smoke run, ~50 steps of `sft-50k`: measure tokens/sec and peak memory, confirm micro-batch 32
      fits, re-estimate run time. (~0.5 CU)
- [x] `sft-50k`, 3 epochs. (~2.5 CU, re-estimated from the smoke run) Held-out 1.4382 / 1.3769 /
      1.3753; ~17 min.
- [x] `sft-2k`, 3 epochs. (~0.5 CU) Held-out 1.9232 / 1.8476 / 1.8342; ~1 min.
- Gates:
  - Held-out loss per epoch for both runs; each run keeps its best epoch.
  - The fixed chat samples in W&B are read by hand at every epoch. A broken mask or template shows
    up there before any metric does.

**Step 5 — Evaluation** (mostly local, 0–3 CU)

```bash
python -m model.domain_eval --checkpoint Juanillaberia/TinyForge-60M-Instruct/sft-2k/epoch_3.pt \
    --preset base60m --tag sft-2k --chat --max-new-tokens 256
python -m model.domain_eval --checkpoint Juanillaberia/TinyForge-60M-Instruct/sft-50k/epoch_3.pt \
    --preset base60m --tag sft-50k --chat --max-new-tokens 256
python -m eval.compare_domain   # -> eval/results/domain-comparison.md
```

On the Mac CPU without a KV cache, both models together took ~5 min (run in parallel; the second
writes to its own `--output` file, appended afterwards).

- [x] Add `chat_prompt` to every item in `eval/domain_prompts.json`.
- [x] Domain eval, appended to `eval/results/domain-answers.jsonl`: tags `sft-50k` and `sft-2k`
      (chat, greedy, 256 new tokens: smol-smoltalk answers are long, and 64-token training samples never reached a stop). Optional `sft-50k-raw`: original completion prompts, no
      template, to check plain completion survived SFT.
- [x] `eval/compare_domain.py` → table per prompt: base (completion prompt, 60 tokens) | `sft-2k` |
      `sft-50k`, plus summary rows:
  - **Stop rate**: % of answers that end in `<|endofturn|>` within the token limit.
  - **Repetition**: fraction of repeated 4-grams in the generated tokens only, on the first 60
    tokens (comparable across the 60- and 256-token limits) and on the whole answer.
  - [ ] **On-topic**: manual y/n per answer, regardless of correctness.
- [ ] lm-eval HellaSwag + ARC-Easy on the chosen SFT checkpoint, to measure the alignment tax vs
      base. Run on the Mac CPU overnight (0 CU), or on the A100 (~2–3 CU) if time matters more.
- [ ] Results into a "Post-training" section of `EXPERIMENTS.md` and a 2.d section of
      `writeup.md`: data stats, held-out loss per epoch for both runs, the domain table, benchmarks,
      and which checkpoint ships as `-Instruct`. Default is `sft-50k`'s best epoch; if `sft-2k`
      matches or beats it, that's the finding and it ships instead.

**Step 6 — Upload to the HF Hub** (Colab CPU runtime, 0 CU)

- [ ] `train/push_to_hub.py`: loads the chosen SFT checkpoint (Hub ref or local path), drops
      optimizer/scaler/RNG state, and writes a weights-only `tinyforge-60m-instruct.pt` (~260 MB).
      It keeps the `model_state_dict` key plus metadata (`preset`, `run_name`, `epoch`,
      `base_checkpoint`, chat format), then uploads it to the root of
      `Juanillaberia/TinyForge-60M-Instruct`.
- [ ] `documentation/model_card_TinyForge-60M-Instruct.md`, pushed as the repo's `README.md`. Same
      structure as the base card: architecture (unchanged), base model link, SFT data and recipe,
      `sft-2k` vs `sft-50k`, base-vs-SFT table, benchmarks, chat format and a usage snippet,
      limitations.
- Gates:
  - In a fresh Colab session, load `Juanillaberia/TinyForge-60M-Instruct/tinyforge-60m-instruct.pt`
    through `load_model_for_eval` and run the domain eval with `--chat`. Output must match the
    Step 5 rows for the chosen checkpoint exactly (greedy).
  - The model card's own usage snippet runs from a fresh clone.
  - The repo stays private until you've reviewed the card; making it public is a manual step.

| Step | Where | CU |
|---|---|---|
| 1–3 template, data, trainer | Mac CPU | 0 |
| 4 smoke + `sft-50k` + `sft-2k` | A100 | ~3.5 |
| 5 domain eval | Mac CPU | 0 |
| 5 lm-eval on SFT | Mac CPU or A100 | 0–3 |
| 6 upload + fresh-session check | Colab CPU | 0 |
| **SFT total** | | **~3.5–6.5** |

### Inference

- [ ] No-cache generation as the baseline (already have this: `model/generate.py`'s
      `greedy_generate`), then a per-layer KV cache. Confirm both produce identical greedy output.
      Needs a start-position argument in `TransformerLM.forward` (RoPE offsets), and a mask that
      works when the query is shorter than the keys.
- [ ] Benchmark tokens/sec with and without the cache, at a few sequence lengths.
- [ ] Add temperature, top-p and repetition-penalty sampling; compare outputs on the same prompts —
      this is also what fixes the repetition loops seen in the base model's domain-eval answers.
- [ ] A CLI chat loop that works from a fresh clone, built on `chat_template.encode_prompt`; Gradio
      if time allows.

### Release and report

- [ ] `TinyForge-60M-Instruct` on the HF Hub (SFT Step 6). Once the KV cache and sampling land,
      update its card's usage snippet to the CLI chat.
- [ ] Write `REPORT.md` per PROJECT_DESIGN §9, with the 2.b experiments as their own section:
      questions, plots, decisions, and how they held up once scaled to `base60m`. Include the
      `sft-2k` vs `sft-50k` ablation the same way.

## Done checklist

- [ ] SFT model answers in chat format; base-vs-SFT table written
- [ ] KV cache matches no-cache output, with a tokens/sec benchmark
- [ ] CLI chat works from a fresh clone
- [ ] `TinyForge-60M-Instruct` on the HF Hub; `REPORT.md` written

## Stretch goals (only after 2.d)

1. High-quality data in a decay phase, tested at small scale first (the SmolLM2 idea) — would need
   its own short experiment before touching `base60m` again.
2. Preference tuning (DPO) on top of SFT, as SmolLM2 does for its Instruct models.
3. Speculative decoding, using the `xs` or `s` model from 2.b as the draft model.
4. RLVR on simple arithmetic, reusing A5.
5. GQA and int8 quantization.
