---
license: mit
language: en
library_name: pytorch
pipeline_tag: text-generation
base_model: Juanillaberia/TinyForge-60M
tags:
  - tinyforge
  - from-scratch
  - small-language-model
  - research
  - sft
datasets:
  - Juanillaberia/TinyForge
  - HuggingFaceTB/smol-smoltalk
---

# TinyForge-60M-Instruct

TinyForge-60M-Instruct is [TinyForge-60M](https://huggingface.co/Juanillaberia/TinyForge-60M), a
60M-parameter decoder-only transformer pretrained from scratch on ~2.5B tokens of FineWeb-Edu, after
supervised fine-tuning (SFT) on 50k chat conversations.

TinyForge is a personal/research project implementing the full LM pipeline end to end — custom
tokenizer, data pipeline, model, trainer, experiments, evaluation, and now post-training — not a
production model. The point of this release is to measure what SFT changes at 60M parameters:
it learns the chat format, when to stop, and to repeat itself far less. It does not learn new
facts; correctness stays bounded by what the base model knows, which is little. Expect a model that
answers in the right shape and is often wrong.

## Model description

Architecture is unchanged from the base model:

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
| Total params | 64,732,800 |

Architecture code: `model/model.py` and `model/components/` in the
[TinyForge repo](https://github.com/JuaniLlaberia/tinyforge).

## Chat format

```
<|endoftext|><|user|>{user message}<|endofturn|><|assistant|>{reply}<|endofturn|>
```

The three chat tokens were reserved in the tokenizer from the start (`<|assistant|>` 257,
`<|user|>` 258, `<|endofturn|>` 259); `<|endoftext|>` is 256. Generation should stop at
`<|endofturn|>`. Multi-turn conversations repeat the user/assistant pair; system prompts, if any,
go at the start of the first user message. `tokenizer/chat_template.py` builds this format for both
training and inference.

## Training data

[`HuggingFaceTB/smol-smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
(built by the SmolLM2 team for small models), a random 50,000-conversation sample of its train
split, plus 31 hand-written research-style conversations: 50,031 conversations, 31.2M tokens.
Conversations longer than the 1,024-token context were truncated at a turn boundary (the longest
prefix of complete user/assistant pairs that fits) rather than dropped, which kept the dataset's
source mix and multi-turn conversations intact; only 2.7% were dropped. Processed splits:
`sft/` in [Juanillaberia/TinyForge](https://huggingface.co/datasets/Juanillaberia/TinyForge).

## Training procedure

| | |
|---|---|
| Initialization | `TinyForge-60M` final checkpoint; chat-token embedding rows reset to the mean embedding (they never occurred in pretraining) |
| Loss | Cross-entropy on assistant replies and their closing `<\|endofturn\|>` only |
| Optimizer | Muon (hidden 2D weight matrices) + AdamW (embeddings/norms), weight decay 0.1 |
| Peak LR | 8e-4 (10x below pretraining) |
| Schedule | 3% warmup, then linear decay to 0 |
| Batch | 64 conversations per step, padded, length-bucketed |
| Epochs | 3 (2,346 steps; held-out loss lowest after epoch 3, but flat from epoch 2) |
| Hardware | 1x A100, ~17 minutes, bf16 |

A same-recipe ablation on a 2k-conversation subset (testing the LIMA claim that ~1k–2k examples
are enough) ended at held-out loss 1.834 vs 1.375 for this model, and never learned to stop.

## Evaluation

**Held-out SFT loss** (1,000 conversations from smol-smoltalk's `test` split, assistant tokens):
2.474 for the base model, 1.375 after SFT.

**Domain eval**: 15 fixed prompts, greedy decoding. The base model continues a text prompt
("The capital of France is"); this model answers the same question in chat form ("What is the
capital of France?"). Base capped at 60 new tokens, this model at 256.

| | TinyForge-60M | TinyForge-60M-Instruct |
|---|---|---|
| Ends its turn on its own | 0/15 | 5/15 |
| Repeated 4-grams in the first 60 tokens | 0.63 | 0.09 |
| On-topic (manual) | 10/15 | 11/15 |
| Correct (12 factual/reasoning/code prompts) | 1/12 | 3/12 |

> **User:** What is the capital of France?
> **TinyForge-60M-Instruct:** The capital of France is Paris. Paris is renowned for its historical
> landmarks, cultural institutions, and intellectual property. It is home to the iconic Eiffel
> Tower, the Louvre Museum, and the Louvre Museum.

> **User:** Write a Python function that reverses a string.
> **TinyForge-60M-Instruct:** Here's a simple Python function that reverses a string:
> `def reverse_string(s): return s[::-1]` (followed by a usage example whose printed output is wrong)

> **User:** What is two plus two?
> **TinyForge-60M-Instruct:** In mathematics, two plus two is a fundamental concept that can be
> expressed as two plus two, two plus two, and two plus two. ... (loops until the token limit)

The base model never names Paris or Jupiter in completion mode; this model does. It still fails
arithmetic, multi-step reasoning, and most code beyond one-liners, and reads "transformer" as the
electrical device.

**Benchmarks** (lm-evaluation-harness, 0-shot, scored as raw text continuations like the base
model, without the chat template):

| | TinyForge-60M | TinyForge-60M-Instruct |
|---|---|---|
| HellaSwag acc / acc_norm | 28.6% / 30.7% | 28.7% / 30.9% |
| ARC-Easy acc / acc_norm | 46.8% / 41.5% | 43.4% / 39.4% |

HellaSwag is unchanged; ARC-Easy drops by a few points, a small cost of SFT (partly, possibly,
because these benchmarks score the base model's text-continuation format).

## How to use

This isn't a `transformers`-compatible checkpoint — it's a plain PyTorch state dict loaded through
the project's own code.

```bash
git clone https://github.com/JuaniLlaberia/tinyforge
cd tinyforge
pip install -r requirements.txt
```

```python
import torch
from config import TokenizerConfig
from model.evaluation import load_model_for_eval
from model.generate import greedy_generate_ids
from tokenizer.chat_template import END_OF_TURN, encode_prompt
from tokenizer.tokenizer_hf import build_hf_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_for_eval("Juanillaberia/TinyForge-60M-Instruct/tinyforge-60m-instruct.pt", "base60m", device)
tokenizer = build_hf_tokenizer()

messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt_ids = encode_prompt(messages, tokenizer)
stop_ids = {tokenizer.token_to_id(END_OF_TURN), TokenizerConfig.eot_id}
new_ids = greedy_generate_ids(model, prompt_ids, context_length=1024, stop_ids=stop_ids, max_new_tokens=256)
print(tokenizer.decode(new_ids))
```

Greedy decoding without a repetition penalty, as in the evaluation above; sampling options and a
CLI chat are planned.

## Limitations

- Not safety-tuned, and no preference tuning (DPO/RLHF) — SFT only.
- Often confidently wrong: weak factual recall, arithmetic and reasoning; code is correct only for
  trivial functions. Do not rely on its answers.
- Still loops on many prompts under greedy decoding, and ends its turn on its own only about a third
  of the time in the domain eval.
- 1,024-token context and 8,192-token vocabulary, both small; long conversations get cut.
- English only. Pretraining data is FineWeb-Edu only; SFT data is synthetic, mostly generated by
  larger models (smol-smoltalk), and inherits their style.

## Citation / acknowledgments

Architecture and process loosely follow the SmolLM2 family, and the SFT data is the SmolLM2 team's
smol-smoltalk. Not affiliated with or endorsed by Hugging Face / the SmolLM2 team.
