from dataclasses import dataclass
from pathlib import Path

@dataclass
class GlobalConfig:
    seed: int = 42

@dataclass
class HFConfig:
    path: str = ""

@dataclass
class DataConfig:
    # Dataset
    source: str = "HuggingFaceFW/fineweb-edu"
    subset: str = "sample-10BT"
    output_path: Path = Path("data/dataset/raw")
    target_tokens: int = 3_000_000_000
    docs_per_chunk: int = 20_000
    keep_columns: tuple = ("id", "text", "url", "dump", "token_count", "score")

    # Train/val split
    train_path: Path = Path("data/dataset/train")
    val_path: Path = Path("data/dataset/val")
    val_split_mod: int = 200

    # unfiltered ablation sample
    raw_sample_path: Path = Path("data/dataset/raw_sample")

    # Preprocessing paths
    preprocessing_output_path: Path = Path("data/dataset/preprocessed")
    filtered_path: Path = Path("data/dataset/preprocessed/filtered")
    exact_dedup_path: Path = Path("data/dataset/preprocessed/exact_dedup")

    # Filters parameters
    doc_min_length: int = 50
    doc_max_length: int = 100_000
    doc_min_mean_length: int = 3
    doc_max_mean_length: int = 10

    deduplication_jaccard_threshold: float = 0.8
    minhash_num_hashes: int = 128
    minhash_num_bands: int = 16
    minhash_ngram_length: int = 5
    language_confidence_filter_threshold: float = 0.7

    # Measurements (Metrics) threshold
    quality_filter_threshold: float = 0.85
    nsfw_filter_threshold: float = 0.75
    hatespeech_filter_threshold: float = 0.75

    # Custom models paths
    nsfw_model_path: str = "data/local_models/nfsw_model.bin"
    hatespeech_model_path: str = "data/local_models/hatespeec_model.bin"
    language_clf_model_path: str = "data/local_models/language_clf.bin"
    quality_clf_model_path: str = "data/local_models/quality_classifier.bin"

@dataclass
class TokenizerConfig:
    vocab_size: int = 8192
    special_tokens: tuple[str, ...] = ("<|endoftext|>", "<|assistant|>", "<|user|>", "<|endofturn|>")

    # Training sample 
    sample_bytes: int = 400_000_000
    sample_path: Path = Path("data/dataset/tokenizer_sample.txt")

    # Final deliverables
    vocab_path: Path = Path("tokenizer/trained_tokenizer/vocab.json")
    merges_path: Path = Path("tokenizer/trained_tokenizer/merges.txt")

    eot_id: int = 256
    shard_tokens: int = 100_000_000
    shards_path: Path = Path("data/dataset/shards")

# Experiments
class XSModelConfig:
    n_layer: int = 6
    d_model: int = 256
    n_head: int = 4
    d_ff: int = 704
    context_length: int = 1024
    theta: float = 10000.0

class SmallModelConfig:
    n_layer: int = 8
    d_model: int = 384
    n_head: int = 6
    d_ff: int = 1024
    context_length: int = 1024
    theta: float = 10000.0

class MidModelConfig:
    n_layer: int = 10
    d_model: int = 512
    n_head: int = 8
    d_ff: int = 1344
    context_length: int = 1024
    theta: float = 10000.0

# Final Base Model Configuration
class Base60MModelConfig:
    n_layer: int = 12
    d_model: int = 640
    n_head: int = 10
    d_ff: int = 1728
    context_length: int = 1024
    theta: float = 10000.0

MODEL_PRESETS = {
    "xs": XSModelConfig,
    "s": SmallModelConfig,
    "m": MidModelConfig,
    "base60m": Base60MModelConfig,
}

class TrainConfig:
    peak_lr_adamw: float = 2e-3
    peak_lr_muon: float = 2e-2
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_frac: float = 0.01
    wsd_decay_frac: float = 0.175
    cosine_min_lr_frac: float = 0.10
    tokens_per_step: int = 256_000
    val_interval: int = 100
    ckpt_interval: int = 500
    num_val_windows: int = 512