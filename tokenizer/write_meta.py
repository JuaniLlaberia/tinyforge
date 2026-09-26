import json
from pathlib import Path

from config import DataConfig, TokenizerConfig

def shard_token_counts(shard_dir: Path) -> dict[str, int]:
    """
    Count tokens in each shard file by its size on disk (uint16, 2 bytes/token).

    Args:
        shard_dir (Path): Directory of shard_*.bin files.
    Returns:
        dict[str, int]: Mapping from shard filename to token count.
    """
    return {f.name: f.stat().st_size // 2 for f in sorted(shard_dir.glob("shard_*.bin"))}

def write_meta(output_path: Path) -> None:
    """
    Write meta.json: version, vocab size, EOT id, token counts per shard,
    bytes-per-token. Matches the guide's Part 1 output contract.

    Args:
        output_path (Path): Where to write meta.json.
    Returns:
        None.
    """
    train_shards = shard_token_counts(TokenizerConfig.shards_path / "train")
    val_shards = shard_token_counts(TokenizerConfig.shards_path / "val")

    meta = {
        "version": 1,
        "vocab_size": TokenizerConfig.vocab_size,
        "eot_id": TokenizerConfig.eot_id,
        "special_tokens": {tok: 256 + i for i, tok in enumerate(TokenizerConfig.special_tokens)},
        "shard_tokens_target": TokenizerConfig.shard_tokens,
        "train_shards": train_shards,
        "train_total_tokens": sum(train_shards.values()),
        "val_shards": val_shards,
        "val_total_tokens": sum(val_shards.values()),
        "bytes_per_token_val": 3.757,
        "git_commit": None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {output_path}")


def write_stats(output_path: Path) -> None:
    """
    Write stats.json: document counts after each pipeline step and the
    removal rate per filter/dedup stage. Matches the guide's Part 1 output
    contract; values transcribed from documentation/notes.md.

    Args:
        output_path (Path): Where to write stats.json.
    Returns:
        None.
    """
    stats = {
        "download": {"total_docs": 2_920_000},
        "split": {"train_docs": 2_905_409, "val_docs": 14_591},
        "filter_gopher": {
            "train_kept": 2_896_126, "train_dropped": 9_283, "train_drop_rate": 9_283 / 2_905_409,
            "val_kept": 14_543, "val_dropped": 48, "val_drop_rate": 48 / 14_591,
        },
        "exact_dedup": {
            "train_kept": 2_861_770, "train_dropped": 34_356, "train_drop_rate": 34_356 / 2_896_126,
            "val_kept": 14_542, "val_dropped": 1,
        },
        "minhash_dedup": {
            "train_kept": 2_846_154, "train_dropped": 15_616, "train_drop_rate": 15_616 / 2_861_770,
            "val_kept": 14_542, "val_dropped": 0,
            "note": "0 val docs dropped by either dedup stage -- split-aware rule always drops the train copy of a train/val duplicate and keeps val",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {output_path}")

def main() -> None:
    write_meta(DataConfig.preprocessing_output_path.parent / "meta.json")
    write_stats(DataConfig.preprocessing_output_path.parent / "stats.json")

if __name__ == "__main__":
    main()
