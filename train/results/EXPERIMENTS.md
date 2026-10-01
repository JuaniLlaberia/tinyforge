# TinyForge - Experiments

## Experiment 0 (seed noise)

Question: How big is seed noise at this scale, so later comparisons know what counts as a real difference?

| Seed | Steps | Tokens seen | Train loss | Val loss | Tokens/sec | Peak mem |
| ---- | ----- | ----------- | ---------- | -------- | ---------- | -------- |
| 42   | 381   | 99,876,864  | 5.0837     | 5.0331   | 256,008    | 4.80 GiB |
| 43   | 381   | 99,876,864  | 5.0046     | 4.9611   | 256,985    | 4.80 GiB |

**Decision:** noise floor established at ≈0.072 nats (≈0.144 nats tie threshold).

---

## Experiment A (AdamW vs Muon)

Question: AdamW or Muon?

| Optimizer | LR   | Train loss | Val loss | Tokens/sec |
| --------- | ---- | ---------- | -------- | ---------- |
| AdamW     | 1e-3 | 5.1302     | 5.0795   | 256,363    |
| AdamW     | 2e-3 | 4.9313     | 4.8759   | 256,085    |
| AdamW     | 4e-3 | 5.1545     | 5.0980   | 256,053    |
| Muon      | 5e-4 | 5.0586     | 5.0087   | 252,224    |
| Muon      | 1e-3 | 4.7747     | 4.7218   | 252,492    |
| Muon      | 2e-3 | 4.3064     | 4.2507   | 252,988    |
| Muon      | 4e-3 | 3.9133     | 3.8647   | 253,114    |

Best AdamW (lr=2e-3, val 4.8759) vs. best Muon (lr=4e-3, val 3.8647): gap ≈ **1.011 nats**,
far beyond the 0.072-nat noise floor from Experiment 0. Muon's val loss improved monotonically
across all 4 tested LRs (5e-4→1e-3→2e-3→4e-3: 5.0087→4.7218→4.2507→3.8647) with no sign of a
ceiling — the true optimum is likely above 4e-3, left unresolved by design (see decision below).

**Decision:** Muon beats AdamW beyond noise, decisively (1.011 nats). Winning optimizer: **Muon**,
lr=4e-3.

## Experiment B (WSD vs cosine)

Question: WSD or cosine?

| Schedule | Train loss | Val loss | Tokens/sec | Final LR                           |
| -------- | ---------- | -------- | ---------- | ---------------------------------- |
| WSD      | 3.4021     | 3.4158   | 251,813    | 3.01e-5 (decayed to ~0)            |
| Cosine   | 3.5705     | 3.5816   | 252,570    | 4.00e-4 (10% of peak, as designed) |

Gap ≈ **0.166 nats** in WSD's favor, somewhat above Experiment 0's raw 0.072-nat floor, but close
to the inflated (~0.157) estimate flagged in Experiment A from likely GPU non-determinism, so I
wouldn't call this an overwhelming margin the way the optimizer decision was.

**Decision:** Winning schedule: **WSD**.

## Experiment C (scaling curve)

Question: is ~60M the right size?

| Preset | Non-emb params | Tokens      | Tokens/param | Train loss | Val loss | Tokens/sec | Peak mem (allocated) |
| ------ | -------------- | ----------- | ------------ | ---------- | -------- | ---------- | -------------------- |
| xs     | 4.82M          | 99,876,864  | 20.7         | 4.1305     | 4.0829   | 437,857    | 3.16 GiB             |
| s      | 14.16M         | 279,969,792 | 19.8         | 3.3089     | 3.3068   | 251,845    | 4.75 GiB             |
| m      | 31.13M         | 624,951,296 | 20.1         | 3.0037     | 3.0253   | 154,828    | 6.89 GiB             |

Fitting a power law (loss ∝ N^-α) across all 3 points on log-log axes gives α ≈ 0.16 — val loss
clearly keeps improving with size at fixed tokens/param, no sign of diminishing returns yet at this
range. Per the guide, three points is a sanity check, not a prediction — this doesn't override the
already-made base60m size decision, just confirms the direction is sound.

**Decision:** Scaling direction confirms base60m is a reasonable choice; no change to the planned.

---

## Final pretraining

Recipe locked in `configs/model.yaml`: Muon, WSD (warmup 1%, decay 17.5%), `base60m` preset, 2.5B
tokens.

### LR sweep at full size

Three ~200-step calibration runs (52,428,800 tokens each) on `base60m`, centered a bit below `s`'s
winning LR per the guide's expectation that the full-size optimum sits lower, then extended upward
since every point kept improving without spiking (same pattern as Experiment A):

| LR   | Val loss (~200 steps) | Gain over previous |
| ---- | --------------------- | ------------------ |
| 4e-3 | 4.3489                | —                  |
| 6e-3 | 4.1868                | +0.162             |
| 8e-3 | 4.1146                | +0.072             |

**Winner: lr=8e-3.**

### Results

| Metric                   | Value                       |
| ------------------------ | --------------------------- |
| Final val loss           | 2.7161                      |
| Val perplexity           | 15.12                       |
| HellaSwag acc / acc_norm | 28.6% / 30.7% (random: 25%) |
| ARC-Easy acc / acc_norm  | 46.8% / 41.5%               |

### Domain eval (base model, pre-SFT)

Fixed 15-prompt set (`eval/domain_prompts.json`) run once, before any SFT work, saved to
`eval/results/domain-answers.jsonl` (tag `base`) to anchor the later base-vs-SFT comparison. Greedy
decoding, no repetition penalty — fluent surface grammar throughout, but frequent repetition loops
and weak factual/arithmetic grounding:

- "The capital of France is" → "...the capital of the French Republic. The capital of the United
  States is Washington, D.C...." (evasive, never names Paris)
- "Two plus two equals" → "...one. Then, the second equals one plus two...." (wrong, degenerates)
- "def fibonacci(n):" → "...- Fibonacci(n):\n- Fibonacci(n):..." (no real code structure)
