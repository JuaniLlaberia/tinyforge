# TinyForge

A small language model built from scratch, end to end: custom tokenizer, data pipeline, transformer
implementation, pretraining trainer, small-scale experiments to pick a recipe, a full pretraining
run, and (next) instruction tuning. Personal/research project, not a production model.

**Status**: Part 1 (data pipeline) and Part 2.a–2.c (components, experiments, pretraining) are done.
`TinyForge-60M` — 12 layers, d_model 640, ~64.7M total params, trained on ~2.5B tokens of
FineWeb-Edu — is being released to the HF Hub now. What's left (SFT, inference, `-Instruct` release)
is tracked in [`documentation/guide-remaining.md`](documentation/guide-remaining.md).

## Repo layout

| Path | What's there |
|---|---|
| `tokenizer/` | Custom BPE tokenizer training, encoding and sharding |
| `data/` | Download, filtering, deduplication pipeline (FineWeb-Edu → uint16 shards) |
| `model/` | Transformer (`model.py`, `components/`), optimizer, scheduler, evaluation and domain-eval CLIs |
| `train/pretrain.py` | The pretraining trainer (checkpoint/resume, W&B logging) |
| `configs/model.yaml` | The locked 2.c recipe (optimizer, schedule, size, token budget) |
| `eval/` | Domain-eval prompt set and results |
| `documentation/` | Per-part writeup, guides, HF model card |
| `train/results/EXPERIMENTS.md` | Small-scale experiment results and decisions (2.b), final pretraining results (2.c) |
| `run_commands.md` | Exact CLI invocations used for every experiment and the final run |

## Running it

```bash
pip install -r requirements.txt
```

**Train** (see `run_commands.md` for every experiment's exact command):
```bash
python -m train.pretrain --preset base60m --optim muon --sched wsd --lr 8e-3 \
    --total-tokens 2_500_000_000 --micro-batch-size 32 --grad-accum-steps 8 \
    --run-name base60m-pretrain --wandb-project tinyforge --hf-repo-id <HF_REPO_ID>
```

**Benchmark eval** (val perplexity + HellaSwag/ARC-Easy via lm-evaluation-harness):
```bash
python -m model.evaluation --checkpoint <HF_REPO_ID>/<run_name>/step_XXXXXXX.pt \
    --preset base60m --tasks hellaswag,arc_easy --output eval/results/base60m_bench.json
```

**Domain eval** (fixed prompt set, saved for base-vs-SFT comparison later):
```bash
python -m model.domain_eval --checkpoint <HF_REPO_ID>/<run_name>/step_XXXXXXX.pt \
    --preset base60m --tag base --output eval/results/domain-answers.jsonl
```

## More detail

- [`documentation/writeup.md`](documentation/writeup.md) — the full technical narrative, Part 1 and Part 2
- [`train/results/EXPERIMENTS.md`](train/results/EXPERIMENTS.md) — every experiment's config, results and decision
- [`documentation/guide-remaining.md`](documentation/guide-remaining.md) — what's left (2.d: SFT, inference, release)
