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

---

## Post-training

### SFT data

Built by `train/prepare_sft_data.py` from `HuggingFaceTB/smol-smoltalk` (seed 42), pushed to
`Juanillaberia/TinyForge` under `sft/`. Full stats: `sft/stats.json`.

Conversations longer than 1,024 input tokens are truncated at a turn boundary (longest prefix of
complete user/assistant pairs that fits) and dropped only if the first pair alone doesn't fit. A
585-conversation smoke run with plain dropping instead lost 49% of conversations and cut 3+-turn
conversations from 59% to 26% (`smol-magpie-ultra-short`, 59% of the source, is mostly 3-turn chats
over 1,024 tokens), so plain dropping was rejected.

| Train pool outcome (51,387 examined) | Count | Share |
|---|---|---|
| Kept whole | 26,348 | 51.3% |
| Truncated at a turn boundary | 23,652 | 46.0% |
| Dropped: first pair too long | 1,384 | 2.7% |
| Dropped: duplicate | 3 | <0.1% |
| Dropped: malformed / special-token text | 0 | 0% |

Truncation is almost entirely `smol-magpie-ultra-short` (77% of its conversations). Drops are mostly
`longalign` (100%, by design of that source) and `smol-summarize-20k` (20%).

| Split | Conversations | Tokens | Mean / median / p95 | Trained-token fraction | Turns 1 / 2 / 3+ |
|---|---|---|---|---|---|
| `train_main` (`sft-50k`) | 50,031 | 31.2M | 623 / 672 / 989 | 79.5% | 47% / 38% / 15% |
| `train_ablation` (`sft-2k`) | 2,031 | 1.24M | 613 / 661 / 987 | 80.1% | 49% / 38% / 13% |
| `heldout` (from `test`) | 1,000 | 0.61M | 609 / 639 / 984 | 79.4% | 51% / 36% / 13% |
| Research prompts (in both train sets) | 31 | 3.7K | 120 / 116 / 187 | 86.4% | 94% / 6% / 0% |

Source mix of `train_main` matches the full train split to within ~1.5 points per source
(`smol-magpie-ultra-short` 60.2% vs 58.8%). No held-out conversation overlaps train or the research
prompts (hash of the rendered conversation).

### SFT runs (`sft-50k` vs `sft-2k`)

Question: does LIMA's small-data claim hold at 60M, i.e. is ~2k conversations enough?

Same recipe for both (`train/sft.py`): base `step_0009535.pt`, chat-token rows initialized to the
mean embedding, Muon + AdamW at peak 8e-4, 3% warmup then linear decay to 0, 64 conversations/step,
3 epochs, loss on assistant tokens only. A100, ~102k tokens/sec (incl. padding), 20 GB peak memory.
Held-out = the same 1,000 `test` conversations for both; base model held-out loss 2.4744.

| Run | Conversations | Steps | Held-out after epoch 1 / 2 / 3 | Assistant tokens trained | Wall time |
| --- | ------------- | ----- | ------------------------------ | ------------------------ | --------- |
| `sft-50k` | 50,031 | 2,346 | 1.4382 / 1.3769 / **1.3753** | 74.3M | ~17 min |
| `sft-2k`  | 2,031  | 96    | 1.9232 / 1.8476 / 1.8342 | 3.0M | ~1 min |

- `sft-50k` beats `sft-2k` by **0.459 nats** on held-out loss, >6x the 0.072-nat seed-noise floor
  from Experiment 0, so one run each is enough to call it.
- Per update, the two runs are about equally efficient: `sft-2k` after 96 steps (1.834, data already
  repeated 3x) is close to `sft-50k` at step 100 (1.775). Repetition didn't hurt; `sft-2k` simply
  got ~24x fewer updates and was still improving when it stopped.
- `sft-50k` plateaued: epoch 3 gained only 0.0016 nats over epoch 2. More epochs on the same 50k
  wouldn't help; more data might.

**Decision (held-out loss):** LIMA's small-data claim does not hold for held-out loss at 60M.
`sft-50k` epoch 3 (`Juanillaberia/TinyForge-60M-Instruct/sft-50k/epoch_3.pt`) is the default
`-Instruct` checkpoint, pending the behavioral comparison in the domain eval (held-out loss
measures fit to smol-smoltalk-style answers, not answer quality).
