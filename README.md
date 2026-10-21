# TinyForge

A small language model built from scratch, end to end: custom tokenizer, data pipeline, transformer
implementation, pretraining trainer, small-scale experiments to pick a recipe, a full pretraining
run, and supervised fine-tuning into a chat model. Personal/research project, not a production
model.

**Status**: Part 1 (data pipeline) and Part 2.a–2.d (components, experiments, pretraining, SFT) are
done. Released on the HF Hub:

| Release | What it is |
|---|---|
| [TinyForge-60M](https://huggingface.co/Juanillaberia/TinyForge-60M) | Base model: 12 layers, d_model 640, ~64.7M params, ~2.5B tokens of FineWeb-Edu. Val perplexity 15.12 |
| [TinyForge-60M-Instruct](https://huggingface.co/Juanillaberia/TinyForge-60M-Instruct) | The base model after SFT on 50k smol-smoltalk conversations. Answers in chat format and stops on its own; still often wrong |
| [Juanillaberia/TinyForge](https://huggingface.co/datasets/Juanillaberia/TinyForge) | Tokenizer, pretraining shards (3.47B tokens) and SFT splits |

Future work (KV-cache inference, sampling, CLI chat, REPORT.md) is tracked in
[`documentation/guide-remaining.md`](documentation/guide-remaining.md).

## Repo layout

| Path | What's there |
|---|---|
| `tokenizer/` | Custom BPE tokenizer training, encoding and sharding; `chat_template.py` (chat format for SFT and inference) |
| `data/` | Download, filtering, deduplication pipeline (FineWeb-Edu → uint16 shards); `data/sft/` hand-written SFT conversations |
| `model/` | Transformer (`model.py`, `components/`), optimizer, scheduler, generation, evaluation and domain-eval CLIs |
| `train/pretrain.py` | The pretraining trainer (checkpoint/resume, W&B logging) |
| `train/prepare_sft_data.py`, `train/sft.py` | SFT data preparation and the SFT trainer |
| `train/push_to_hub.py` | Strips a checkpoint to weights only and publishes it with its model card |
| `configs/model.yaml` | The locked 2.c pretraining recipe (optimizer, schedule, size, LR, token budget) |
| `eval/` | Domain-eval prompts, comparison script and results (benchmarks, answers, judgments) |
| `documentation/` | Writeup, guides, HF model and dataset cards |
| `train/results/EXPERIMENTS.md` | Every experiment's config, results and decision (2.b, 2.c, 2.d) |

## Running it

```bash
pip install -r requirements.txt
```

**Pretrain:**
```bash
python -m train.pretrain --preset base60m --optim muon --sched wsd --lr 8e-3 \
    --total-tokens 2_500_000_000 --micro-batch-size 32 --grad-accum-steps 8 \
    --run-name base60m-pretrain --wandb-project tinyforge --hf-repo-id <HF_REPO_ID>
```

**SFT** (data is pulled from the HF dataset repo if not built locally):
```bash
python -m train.prepare_sft_data --hf-repo-id <HF_DATASET_REPO_ID>   # optional: rebuild the SFT splits
python -m train.sft --run-name sft-50k --hf-repo-id <HF_REPO_ID>      # or sft-2k for the ablation
python -m train.push_to_hub --checkpoint <HF_REPO_ID>/sft-50k/epoch_3.pt --run-name sft-50k
```

**Benchmark eval** (val perplexity + HellaSwag/ARC-Easy via lm-evaluation-harness;
`--skip-perplexity` if the pretraining shards aren't on disk):
```bash
python -m model.evaluation --checkpoint <HF_REPO_ID>/<run_name>/<file>.pt \
    --preset base60m --tasks hellaswag,arc_easy --output eval/results/<name>_bench.json
```

**Domain eval** (fixed 15-prompt set; `--chat` for SFT models) and the base-vs-SFT comparison:
```bash
python -m model.domain_eval --checkpoint <HF_REPO_ID>/<run_name>/<file>.pt \
    --preset base60m --tag sft-50k --chat --max-new-tokens 256
python -m eval.compare_domain   # -> eval/results/domain-comparison.md
```

**Chat with the Instruct model** (greedy, from the repo root): see the usage snippet in the
[TinyForge-60M-Instruct card](https://huggingface.co/Juanillaberia/TinyForge-60M-Instruct).

## More detail

- [`documentation/writeup.md`](documentation/writeup.md) — the full technical narrative, Part 1 and Part 2 (2.a–2.d)
- [`train/results/EXPERIMENTS.md`](train/results/EXPERIMENTS.md) — every experiment's config, results and decision
- [`eval/results/domain-comparison.md`](eval/results/domain-comparison.md) — base vs SFT answers, side by side
- [`documentation/guide-remaining.md`](documentation/guide-remaining.md) — what's done and future work
