# TinyForge — Part 2: Model, Training + Post-Training Guide

Sep 25, 2026 · @Juan Ignacio Llaberia

## Purpose and starting point

Part 2 turns the Part 1 shards into a pretrained \~60M-parameter model, an SFT chat version, a KV-cached inference demo, and REPORT.md. Plan on about 5 weeks at part-time pace.

It starts from the output contract in the Part 1 guide: tokenizer files, uint16 train/val shards, and `meta.json` on the HF Hub. Nothing in Part 2 reads raw text, so if the shards load and decode correctly, you can begin.

The architecture is a \~60M-parameter decoder: 12 layers at d=640. It replaces the 8-layer, d=512 small tier in PROJECT\_DESIGN §2, which works out to only \~25M non-embedding parameters (each layer is about 12·d²).

| Component | Choice |
| --- | --- |
| Layers | 12 |
| Hidden dim | 640 |
| Heads | 10 (head dim 64) |
| MLP | SwiGLU, hidden 1,728 |
| Norm | RMSNorm, pre-norm |
| Positions | RoPE, causal masking |
| Context length | 1,024 |
| Vocab | 8,192, tied input/output embeddings |
| Params | \~59.5M non-embedding + \~5.2M embedding |

At 2.5B tokens this is \~40 tokens per parameter, about 2× Chinchilla-optimal.

## Phase roadmap

&#91;embedded content: Part 2 roadmap · 5 phases, 4 gates\]

Pretraining is the long pole and runs in the background across many Colab sessions. You can write eval and SFT code while it runs, but don't evaluate or fine-tune until its gate passes.

## Phase details

### Phase 1 — Model and training loop (week 1)

1. Port your A1 transformer into `model/model.py` with the 12-layer config above. A1 already has RoPE, RMSNorm and SwiGLU; add weight tying and use `F.scaled_dot_product_attention` for the attention kernel.
2. Write a memmap data loader over the uint16 shards: random 1025-token windows, returning inputs and next-token targets.
3. Build `train/pretrain.py`: AdamW (betas 0.9/0.95, weight decay 0.1), grad clip 1.0, gradient accumulation, WSD schedule, W&B logging, and checkpoint/resume from day one.
4. Precision depends on the GPU: bf16 on A100 or L4, fp16 with a GradScaler on T4 (T4 has no bf16). Detect it at startup.
5. Gate checks: loss goes near zero on a single repeated batch, and a run killed mid-way resumes to the identical step, LR and loss.

### Phase 2 — Pretraining (weeks 2–3)

1. Run a short LR sweep first: 3 runs of \~200 steps at peak LRs around 3e-4, 6e-4 and 1e-3. Keep the one with the lowest loss that doesn't spike.
2. Aim for \~256K tokens per optimizer step (for example micro-batch 16 × 1024 tokens × 16 accumulation steps). At 2.5B tokens that is roughly 9,500 steps.
3. WSD: warm up over the first \~1% of steps, hold the peak LR, and decay over the last \~15%. Because the stable phase is flat, you can decide how long to train while it runs.
4. Log val loss every \~250 steps and generate 3 fixed prompts at each checkpoint. Reading samples catches bugs that loss curves hide.

### Phase 3 — Evaluation (week 4)

1. Report final val loss and perplexity on the held-out shard.
2. LM Eval Harness needs a small adapter: subclass its `LM` class and implement `loglikelihood` with your model and tokenizer. Run HellaSwag and ARC-Easy.
3. Write the 10–20 prompt domain eval now, before SFT, so you can compare base and SFT on the same prompts.
4. Expect near-random benchmark scores at 60M parameters. State that up front in the report.

### Phase 4 — SFT (week 4)

1. Build a LIMA-style set of 1,000–2,000 examples. `HuggingFaceTB/smol-smoltalk` was built for small models and is a good base; add a few dozen of your own research-style prompts.
2. Format with the reserved chat tokens from your tokenizer, and mask the loss on everything except assistant turns (A5 code).
3. Use a much lower LR than pretraining (around 10× lower) for 2–3 epochs, and track held-out SFT loss.
4. Rerun the domain eval and put base vs SFT answers side by side.

### Phase 5 — Inference, demo and write-up (week 5)

1. Implement no-cache generation as the baseline, then the per-layer KV cache. Check that both produce identical greedy output, then benchmark tokens/sec at several sequence lengths.
2. Add temperature, top-p and repetition-penalty sampling; compare outputs on the same prompts.
3. Build the CLI chat loop (Gradio if time allows).
4. Write REPORT.md following PROJECT\_DESIGN §9, and push the final checkpoints to the HF Hub.

## Colab run discipline

Treat every session as one that can die at any minute; the run should survive that without losing more than one checkpoint interval.

- **What a checkpoint holds.** Model, optimizer state, scaler (fp16), step, data-loader RNG state, W&B run ID. Missing any of these means the resumed run isn't the same run.
- **Where it goes.** Push to the HF Hub or Drive every \~30 minutes of training, keeping the last 2 plus milestone checkpoints. Never only `/content`.
- **Session start ritual.** Run `nvidia-smi`, pick micro-batch and precision for that GPU, pull the latest checkpoint, resume the same W&B run.
- **Budget.** Per PROJECT\_DESIGN §5, 100 CU is about 57 T4-hours or 7 A100-hours. Log tokens/sec per GPU type in week 1 and redo the math with real numbers before committing to 2.5B tokens.
- **Don't compare across GPUs by step count.** Use tokens processed as the x-axis in W&B, so runs on different GPUs line up.

## Stretch goals and done checklist

Stretch goals, in the order from PROJECT\_DESIGN §11, only after the checklist below is complete:

1. Mini scaling-law check: the same recipe at 2–3 smaller sizes, plotted like A3. The smallest model doubles as a draft model for speculative decoding.
2. Classification head on the final hidden state.
3. RLVR on simple arithmetic, reusing A5.
4. GQA, int8 quantization.

Done when:

- [ ] Resume tested before the long run started
- [ ] Pretraining finished with train and val curves in W&B
- [ ] Val perplexity and LM Eval Harness numbers recorded
- [ ] SFT model answers in chat format; base vs SFT comparison table written
- [ ] KV cache matches no-cache output and has a tokens/sec benchmark
- [ ] CLI chat demo works from a fresh clone
- [ ] REPORT.md written; checkpoints on the HF Hub
