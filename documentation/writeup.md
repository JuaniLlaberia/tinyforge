# TinyForge Notes: Part 1 (Data Pipeline)

## Dataset

### Download

We used the `HuggingFaceFW/fineweb-edu` dataset, `sample-10BT` subset, with an approximate budget of 3B tokens.

A custom downloader fetched the data in chunks and stored it as parquet files. It took 9 minutes 19 seconds (5,345,689.5 tokens/s).

The data was split into training and validation sets:

| Split | Docs |
|---|---|
| Train | 2,905,409 |
| Val | 14,591 |

### Preprocessing

The dataset is the "edu" version, which already has some filtering and cleanup applied, but we ran additional filters on top.

**Gopher filter**

| Split | Kept | Dropped |
|---|---|---|
| Train | 2,896,126 | 9,283 |
| Val | 14,543 | 48 |

The quality filter (custom model) was skipped to save compute, since the source dataset is already high quality. A random sample was set aside for future ablation experiments.

**Deduplication**

We deduplicated in two passes:

| Method | Train | Val |
|---|---|---|
| Exact dedup | 2,861,770 | 14,542 |
| MinHash dedup | 2,846,154 | 14,542 |

### Results

Train count through the pipeline:

| Stage | Train docs | Change |
|---|---|---|
| Post-split | 2,905,409 | |
| Gopher filter | 2,896,126 | -9,283 |
| Exact dedup | 2,861,770 | -34,356 |
| MinHash near-dedup | 2,846,154 | -15,616 |

Val stayed at 14,542 through both dedup stages. Zero val docs were dropped, because the split-aware dedup rule always drops the train copy of a train/val duplicate and keeps val untouched. This is what prevents train/val leakage.

## Tokenizer

### Training

Trained on a 400MB random sample of the filtered and deduped train text (83,639 docs, seed 42).

| Metric | Value |
|---|---|
| Merges | 7,937 |
| Vocab size | 8,192 / 8,192 |
| Corpus preprocessing | 60.2s |
| BPE training | 147.5s |
| Total | 2m 27s |

Output saved to `tokenizer/vocab.json` and `tokenizer/merges.txt`.

### Encoding and sharding

Used HF's `encode_batch()` per chunk. This ran at roughly 650% CPU via native Rust thread parallelism (no explicit multiprocessing needed), at about 3,650 to 4,700 docs/s.

| Split | Tokens | Shards | Size |
|---|---|---|---|
| Train | 3,470,809,194 | 35 (34 full @ 100M + 1 partial @ 70,809,194) | 6.5GB |
| Val | 17,403,724 | 1 | 33MB |

Total elapsed: 13m 5s (13m 1s train, 3s val).

Note: EOT id is 256 (`special_tokens[0]`, right after the 256 raw bytes). This is not the "usually last vocab id" convention some guides use, noted here to avoid ambiguity in Part 2.

### Upload and verification

Uploaded to the private HF Hub dataset repo `Juanillaberia/TinyForge`: train shards, val shard, `meta.json`, `stats.json`. 41 files total, about 6.5GB.

Verified by downloading `vocab.json`, `merges.txt`, `meta.json`, and 3 shards (2 train, 1 val) fresh from the Hub into a scratch dir, building a tokenizer purely from those downloaded files, and decoding 5 random 400-token windows. All were real, coherent FineWeb-Edu text.

`meta.json` read back correctly: vocab_size 8192, train 3,470,809,194 tokens, val 17,403,724 tokens.

## Where We Are Now

Part 1 (data pipeline) is complete. The final dataset lives at [Juanillaberia/TinyForge](https://huggingface.co/datasets/Juanillaberia/TinyForge) on the HF Hub, verified end to end.

Part 2 (pretraining, on Colab GPU) starts from that Hub dataset and does not touch raw text again. Everything Part 2 needs:

- Tokenizer files: `vocab.json`, `merges.txt`
- Tokenized shards: 35 train shards, 1 val shard
- `meta.json` (vocab size, token counts) and `stats.json`
- EOT token id is 256, not the last vocab id

Open item carried over: the quality filter was not run, and a random sample was set aside for possible ablation experiments later.
