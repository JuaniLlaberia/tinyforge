---
license: odc-by
language: en
tags:
  - tinyforge
  - pretraining
  - sft
  - fineweb-edu
size_categories:
  - 1B<n<10B
---

# TinyForge data

Training data for [TinyForge-60M](https://huggingface.co/Juanillaberia/TinyForge-60M) and
[TinyForge-60M-Instruct](https://huggingface.co/Juanillaberia/TinyForge-60M-Instruct), a
from-scratch small language model built as a personal research project. Code that produced every
file here: [github.com/JuaniLlaberia/tinyforge](https://github.com/JuaniLlaberia/tinyforge).

## Contents

| Path | Contents |
|---|---|
| `tokenizer/vocab.json`, `tokenizer/merges.txt` | Custom byte-level BPE tokenizer, 8,192 tokens including 4 special tokens |
| `train/shard_0000.bin` … `shard_0034.bin` | Pretraining tokens, uint16, 100M per shard (last shard partial): 3,470,809,194 tokens |
| `val/shard_0000.bin` | Held-out validation tokens, same format: 17,403,724 tokens |
| `meta.json` | Vocab size, special-token ids, token count per shard |
| `stats.json` | Document counts after each pipeline step |
| `sft/train_main.jsonl` | SFT set used for TinyForge-60M-Instruct: 50,031 chat conversations |
| `sft/train_ablation.jsonl` | 2,031-conversation subset of the above, for the small-data ablation |
| `sft/heldout.jsonl` | 1,000 held-out conversations for SFT loss |
| `sft/stats.json` | SFT filtering, truncation, token and source-mix stats |

## Pretraining tokens

Source: [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
`sample-10BT` subset, ~2.9M documents. Pipeline:

| Stage | Train docs | Val docs |
|---|---|---|
| Downloaded and split (0.5% to val by hash of doc id, before any filtering) | 2,905,409 | 14,591 |
| Gopher-style quality filter | 2,896,126 | 14,543 |
| Exact dedup | 2,861,770 | 14,542 |
| MinHash near-dedup | 2,846,154 | 14,542 |

Dedup ran across train and val together, and whenever a duplicate spanned both splits the train
copy was dropped, so no validation text leaks into training. Documents are encoded in pipeline order
(not shuffled), each followed by `<|endoftext|>`; training draws random windows across all shards,
so shard order doesn't affect it.

Shards are raw `uint16` token ids: read with `numpy.memmap(path, dtype=numpy.uint16, mode="r")`.
Special token ids: `<|endoftext|>` 256, `<|assistant|>` 257, `<|user|>` 258, `<|endofturn|>` 259
(256 raw bytes come first, so `<|endoftext|>` is not the last id).

## SFT conversations

Source: [`HuggingFaceTB/smol-smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
(Apache-2.0), plus 31 hand-written research-style conversations (source `tinyforge-research`).
Each line is `{"messages": [{"role": "user"|"assistant", "content": ...}, ...], "source", "num_tokens",
"num_trained_tokens", "num_turns"}`.

- System messages are merged into the first user turn.
- Conversations longer than 1,024 tokens (in this tokenizer) are truncated at a turn boundary, keeping
  the longest prefix of complete user/assistant pairs that fits; dropped only if the first pair
  alone doesn't fit (2.7%).
- `train_main` is a random sample (seed 42) of smol-smoltalk's train split; `train_ablation` is its
  first 2,000 conversations plus the same 31 hand-written ones; `heldout` comes from
  smol-smoltalk's `test` split, with no overlap with either train set.

## License

The pretraining shards and tokenizer derive from FineWeb-Edu (ODC-By 1.0); the `sft/` files derive
from smol-smoltalk (Apache-2.0). Use them under the terms of those sources.
