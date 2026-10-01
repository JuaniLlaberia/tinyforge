# TinyForge — what's left

Oct 1, 2026 · @Juan Ignacio Llaberia

Replaces `guide-part2.md` and `guide-part2-revised.md`, both now executed through 2.c. This guide
covers only what's still open: finishing 2.c's release, and all of 2.d.

Full history of how the recipe was chosen lives in `EXPERIMENTS.md`; the Part 2 pretraining results
(val perplexity 15.12, HellaSwag/ARC-Easy, domain-eval baseline) are written up in
`documentation/writeup.md`.

## Finish 2.c — release TinyForge-60M

- [ ] Fill in the remaining placeholders in `documentation/model_card_TinyForge-60M.md`
      (`<GITHUB_REPO_URL>`, if still present) and push it as `README.md` to the
      `Juanillaberia/TinyForge-60M` HF repo root, alongside the final checkpoint.
- [ ] Confirm the model loads cleanly from a fresh clone using the model card's own usage snippet.

## 2.d Post-training and release → TinyForge-60M-Instruct

Plan about 1–2 weeks part-time. ~5% of the 100 CU budget; SFT is small, inference benchmarks are
cheap. Keep the ~10% overall reserve untouched unless something crashes.

**SFT**

- [ ] Build a LIMA-style set of 1,000–2,000 examples: `HuggingFaceTB/smol-smoltalk` (built by the
      SmolLM2 team for small models) plus a few dozen original research-style prompts.
- [ ] Format with the tokenizer's reserved chat tokens (`<|user|>`, `<|assistant|>`,
      `<|endofturn|>`); mask the loss on everything except assistant turns.
- [ ] Train at ~10x lower LR than pretraining (so ~8e-4 off this run's 8e-3 peak) for 2–3 epochs,
      tracking held-out SFT loss; stop at the epoch where it bottoms out.
- [ ] Rerun the domain eval (`model/domain_eval.py --tag sft`, same `eval/domain_prompts.json`,
      appended to the existing `eval/results/domain-answers.jsonl`) and write a base-vs-SFT table:
      same prompts, both answers side by side.

**Inference**

- [ ] No-cache generation as the baseline (already have this: `model/generate.py`'s
      `greedy_generate`), then a per-layer KV cache. Confirm both produce identical greedy output.
- [ ] Benchmark tokens/sec with and without the cache, at a few sequence lengths.
- [ ] Add temperature, top-p and repetition-penalty sampling; compare outputs on the same prompts —
      this is also what fixes the repetition loops seen in the base model's domain-eval answers.
- [ ] A CLI chat loop that works from a fresh clone; Gradio if time allows.

**Release and report**

- [ ] Push `TinyForge-60M-Instruct` to the HF Hub with its own model card (same structure as
      `model_card_TinyForge-60M.md`: architecture — unchanged from base — training data, SFT
      recipe, base-vs-SFT table, limitations).
- [ ] Write `REPORT.md` per PROJECT_DESIGN §9, with the 2.b experiments as their own section:
      questions, plots, decisions, and how they held up once scaled to `base60m`.

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
