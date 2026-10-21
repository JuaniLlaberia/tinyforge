---
license: mit
language: en
library_name: pytorch
pipeline_tag: text-generation
tags:
  - tinyforge
  - from-scratch
  - small-language-model
  - research
datasets:
  - Juanillaberia/TinyForge
---

# TinyForge-60M

TinyForge-60M is a 60M-parameter decoder-only transformer, pretrained from scratch on ~2.5B tokens
of FineWeb-Edu. It's a base (non-instruction-tuned) model; its chat version,
[TinyForge-60M-Instruct](https://huggingface.co/Juanillaberia/TinyForge-60M-Instruct), adds
supervised fine-tuning on top of this checkpoint.

TinyForge is a personal/research project implementing the full LM pipeline end to end — custom
tokenizer, data pipeline, model, trainer, optimizer/schedule experiments, and evaluation — not a
production model. SmolLM2 (the nearest public reference at a similar recipe family) is roughly 2x
the parameters and trained on three orders of magnitude more tokens; treat any comparison to it as
a check on process and curve shape, not a performance target.

## Model description

| | |
|---|---|
| Layers | 12 |
| Hidden size (d_model) | 640 |
| Attention heads | 10 (head dim 64) |
| Feed-forward | SwiGLU, d_ff 1,728 |
| Normalization | RMSNorm, pre-norm |
| Position encoding | RoPE (theta 10,000) |
| Context length | 1,024 tokens |
| Vocabulary | 8,192 (custom BPE, tied embedding/output head) |
| Non-embedding params | 59,489,920 |
| Embedding params | 5,242,880 |
| Total params | 64,732,800 |

Architecture code: `model/model.py` and `model/components/` in the
[TinyForge repo](https://github.com/JuaniLlaberia/tinyforge).

## Training data

Pretrained on a FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, `sample-10BT`) derivative, filtered and
deduplicated, tokenized with a custom 8,192-vocab byte-level BPE tokenizer trained on the project's
own data. Full pipeline and stats: Gopher-quality-filtered, then exact + MinHash deduplicated, down
to 2,846,154 train documents / 3,470,809,194 train tokens available across 35 shards (this run used
2.5B of those). Dataset and tokenizer artifacts: [Juanillaberia/TinyForge](https://huggingface.co/datasets/Juanillaberia/TinyForge).

EOT token id is 256 (not the last vocab id — first special token right after the 256 raw byte
tokens).

## Training procedure

Recipe locked from small-scale experiments (`s`/`xs`/`m` presets, see the repo's
`train/results/EXPERIMENTS.md`) before committing compute to this run:

| | |
|---|---|
| Optimizer | Muon (hidden 2D weight matrices) + AdamW (embeddings/head/norms) |
| Schedule | WSD — warmup 1%, flat peak, linear decay over the last 17.5% |
| Peak LR | 8e-3, from ~200-step full-size calibration runs at 4e-3 / 6e-3 / 8e-3 (val loss 4.349 / 4.187 / 4.115), extended upward while each step kept improving without a loss spike |
| Tokens/step | 262,144 (micro-batch 32 x grad-accum 8 x 1,024 context) |
| Total tokens | ~2.5B (9,535 steps) |
| Precision | bf16 autocast (A100/L4) or fp16 + GradScaler (T4), detected at runtime |
| Grad clip | 1.0 |

Optimizer and schedule were decided by direct comparison at small scale: Muon beat AdamW by 1.011
nats (vs. a ~0.072-nat seed-noise floor), and WSD beat cosine by 0.166 nats, both at the `s` preset
(8 layers, d_model 384). A 3-point scaling check (xs/s/m, ~20 tokens/param) confirmed the loss curve
keeps improving with size in this range before committing to base60m.

## Evaluation

| Metric | Value |
|---|---|
| Val loss | 2.7161 |
| Val perplexity | 15.12 |
| HellaSwag acc / acc_norm | 28.6% / 30.7% (random: 25%) |
| ARC-Easy acc / acc_norm | 46.8% / 41.5% |

Benchmarked via [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(0-shot) through a custom adapter around TinyForge's own tokenizer (`model/evaluation.py`). Both
scores sit modestly above their random baselines — real but limited signal, consistent with a 60M
model trained on a comparatively small token budget. Val loss/perplexity use the project's fixed,
held-out validation shard.

### Domain-eval samples (greedy decoding, no repetition penalty)

A fixed 15-prompt set (`eval/domain_prompts.json`), run before any SFT to anchor the base-vs-SFT
comparison. A few representative completions:

> **Prompt:** "The capital of France is"
> **Completion:** "The capital of France is the capital of the French Republic. The capital of the United States is Washington, D.C. ..."

> **Prompt:** "def fibonacci(n):"
> **Completion:** "def fibonacci(n):\n- Fibonacci(n):\n- Fibonacci(n):\n- Fibonacci(n): ..."

> **Prompt:** "Two plus two equals"
> **Completion:** "Two plus two equals one.\nThen, the second equals one plus two. ..."

Fluent surface grammar, but weak factual/arithmetic grounding and frequent repetition loops. Full
set: `eval/results/domain-answers.jsonl` in the repo; the comparison against the SFT model is in
`eval/results/domain-comparison.md` and the
[TinyForge-60M-Instruct](https://huggingface.co/Juanillaberia/TinyForge-60M-Instruct) card.

## How to use

This isn't a `transformers`-compatible checkpoint — it's a plain PyTorch state dict loaded through
the project's own loader.

```bash
git clone https://github.com/JuaniLlaberia/tinyforge
cd tinyforge
pip install -r requirements.txt
```

```python
import torch
from model.evaluation import load_model_for_eval
from model.generate import greedy_generate
from tokenizer.tokenizer_hf import build_hf_tokenizer
from config import TokenizerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_for_eval(
    "Juanillaberia/TinyForge-60M/base60m-pretrain/step_0009535.pt",
    "base60m",
    device,
)
tokenizer = build_hf_tokenizer()

print(greedy_generate(model, tokenizer, "The history of the Roman Empire began",
                       context_length=1024, eot_id=TokenizerConfig.eot_id, max_new_tokens=60))
```

## Limitations

- Base model only — not instruction-tuned, not safety-tuned. Expect base-LM behavior: repetition
  loops under greedy decoding, weak factual/arithmetic reliability, no instruction-following. Use
  [TinyForge-60M-Instruct](https://huggingface.co/Juanillaberia/TinyForge-60M-Instruct) for chat.
- Trained on ~2.5B tokens, about 40 tokens per parameter: roughly 2x the ~20 tokens/parameter
  Chinchilla rule of thumb for compute-optimal training. This deliberately spends extra compute for
  a smaller, cheaper-to-run model at a given loss, the same trade SmolLM2 makes (far more
  aggressively).
- 8,192-token vocabulary and 1,024-token context are both small relative to typical modern LMs.
- Trained on FineWeb-Edu only; no code-, math-, or dialogue-specific data mixed in, which shows in
  the domain-eval samples above.

## Citation / acknowledgments

Architecture and process loosely follow the SmolLM2 family (naming convention: base model +
`-Instruct`). Not affiliated with or endorsed by Hugging Face / the SmolLM2 team.
