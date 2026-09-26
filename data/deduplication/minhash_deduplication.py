import random
import re
import time
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import xxhash

from config import GlobalConfig, DataConfig
from data.progress import format_duration
from data.split_dataset import write_parquet

# 2**31-1 (not 2**61-1): keeps a*h products inside int64 range so
# calculate_minhash can be vectorized with numpy instead of a pure-Python
# nested loop. Still astronomically larger than any realistic corpus size,
# so collision risk is unaffected in practice -- this is purely an
# implementation-scale choice, not a change to dedup quality.
MERSENNE_PRIME = (1 << 31) - 1

# Combining diacritical marks, the block accented Latin characters decompose
# into after NFD normalization. A single regex sub here replaces what was a
# pure-Python per-character unicodedata.combining() loop -- the same result,
# but at C-level regex speed instead of ~billions of Python function calls
# across a multi-million-document corpus.
_COMBINING_MARKS = re.compile(r"[̀-ͯ]")

def normalize_text(text: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace, and remove accents.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text.
    """
    if not text:
        return ""

    text = " ".join(text.split()).lower()
    text = re.sub(r"[^\w\s]", "", text)
    normalized = unicodedata.normalize("NFD", text)

    return _COMBINING_MARKS.sub("", normalized)

def generate_ngrams(text: str, n: int) -> list[str]:
    """
    Split text into word n-grams.

    Args:
        text (str): Input text.
        n (int): N-gram length in words.

    Returns:
        list[str]: List of n-grams, each joined back into a single string.
    """
    words = text.split()

    if len(words) < n:
        return [text] if text else []

    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]

def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Compute the Jaccard similarity between two sets.

    Args:
        set1 (set): First set.
        set2 (set): Second set.

    Returns:
        float: Jaccard similarity in [0, 1].
    """
    if not set1 and not set2:
        return 1.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union else 0.0

def stable_ngram_hash(ngram: str) -> int:
    """
    Compute a deterministic hash for an n-gram, stable across processes/runs.

    Args:
        ngram (str): N-gram string.

    Returns:
        int: 64-bit integer hash.
    """
    return xxhash.xxh64(ngram.encode("utf-8")).intdigest()

def generate_hash_functions(num_hashes: int, seed: int = GlobalConfig.seed) -> np.ndarray:
    """
    Generate independent (a, b) coefficient pairs for universal hashing.

    Args:
        num_hashes (int): Number of hash functions to generate.
        seed (int): Random seed, so signatures are reproducible and comparable across documents.

    Returns:
        np.ndarray: Shape (num_hashes, 2) int64 array of (a, b) coefficient
        pairs, where h(x) = (a * x + b) mod MERSENNE_PRIME.
    """
    rng = random.Random(seed)
    pairs = [
        (rng.randint(1, MERSENNE_PRIME - 1), rng.randint(0, MERSENNE_PRIME - 1))
        for _ in range(num_hashes)
    ]
    return np.array(pairs, dtype=np.int64)

def calculate_minhash(ngrams: set[str], hash_functions: np.ndarray) -> np.ndarray:
    """
    Compute the MinHash signature for a set of n-grams, vectorized with numpy.

    Args:
        ngrams (set[str]): Set of n-grams representing a document.
        hash_functions (np.ndarray): Shape (num_hashes, 2) array from `generate_hash_functions`.

    Returns:
        np.ndarray: MinHash signature, shape (num_hashes,), one value per hash function.
    """
    num_hashes = hash_functions.shape[0]
    if not ngrams:
        return np.zeros(num_hashes, dtype=np.int64)

    base_hashes = np.fromiter(
        (stable_ngram_hash(g) % MERSENNE_PRIME for g in ngrams), dtype=np.int64, count=len(ngrams)
    )

    a = hash_functions[:, 0:1]  # (num_hashes, 1)
    b = hash_functions[:, 1:2]  # (num_hashes, 1)
    hashed = (a * base_hashes[None, :] + b) % MERSENNE_PRIME  # (num_hashes, num_ngrams)

    return hashed.min(axis=1)

def band_signature(signature, num_bands: int) -> list[tuple]:
    """
    Split a MinHash signature into equal-length bands.

    Args:
        signature: MinHash signature (list or numpy array of ints).
        num_bands (int): Number of bands to split the signature into.

    Returns:
        list[tuple]: List of band tuples, hashable and comparable across documents.
    """
    r = len(signature) // num_bands
    return [tuple(int(x) for x in signature[i * r:(i + 1) * r]) for i in range(num_bands)]

def band_slice(signature, band_idx: int, band_size: int) -> tuple:
    """
    Extract a single band (as a hashable tuple) from a MinHash signature.

    Used instead of `band_signature` when bands are processed one at a time
    -- computing only the requested band avoids building bucket dicts for
    all bands simultaneously, which at a few million documents is the
    difference between ~O(total_docs) and ~O(total_docs * num_bands) of
    live Python objects.

    Args:
        signature: MinHash signature (numpy array of ints).
        band_idx (int): Which band to extract.
        band_size (int): Hash values per band (num_hashes // num_bands).

    Returns:
        tuple: The band's values as a hashable tuple.
    """
    start = band_idx * band_size
    return tuple(int(x) for x in signature[start:start + band_size])

def find_candidate_pairs(signatures: np.ndarray, num_bands: int) -> set[tuple[int, int]]:
    """
    Find LSH candidate pairs, processing one band at a time so only one
    band's bucket dict is ever resident (not all `num_bands` simultaneously).

    Args:
        signatures (np.ndarray): Shape (num_docs, num_hashes) signature matrix.
        num_bands (int): Number of LSH bands (must evenly divide num_hashes).

    Returns:
        set[tuple[int, int]]: Candidate document-index pairs sharing at least one band.
    """
    total_docs, num_hashes = signatures.shape
    band_size = num_hashes // num_bands

    candidate_pairs: set[tuple[int, int]] = set()
    for band_idx in range(num_bands):
        bucket: dict[tuple, list[int]] = defaultdict(list)
        for doc_idx in range(total_docs):
            bucket[band_slice(signatures[doc_idx], band_idx, band_size)].append(doc_idx)

        for doc_list in bucket.values():
            for i in range(len(doc_list)):
                for j in range(i + 1, len(doc_list)):
                    candidate_pairs.add(tuple(sorted((doc_list[i], doc_list[j]))))

        print(f"MinHash banding | band {band_idx + 1}/{num_bands} | {len(candidate_pairs)} candidate pairs so far")

    return candidate_pairs

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        self.parent[self.find(x)] = self.find(y)

def minhash_deduplication(
    paths: list[Path],
    num_hashes: int,
    num_bands: int,
    ngram_length: int,
    output_dir: Path,
    jaccard_threshold: float = DataConfig.deduplication_jaccard_threshold,
    seed: int = GlobalConfig.seed,
) -> Path:
    """
    Remove near-duplicate documents using MinHash + LSH banding.

    Args:
        paths (list[Path]): Input document files.
        num_hashes (int): Number of hash functions in each MinHash signature (Must be evenly divisible by `num_bands`).
        num_bands (int): Number of LSH bands to split each signature into.
        ngram_length (int): Word n-gram length used to represent each document.
        output_dir (Path): Directory to write surviving documents to.
        jaccard_threshold (float): Minimum true Jaccard similarity to treat a candidate pair as a duplicate.
        seed (int): Random seed for hash function generation and survivor choice.

    Returns:
        The output directory containing the deduplicated documents.
    """
    if num_hashes % num_bands != 0:
        raise ValueError("Number of hashes must be evenly divisible by the number of bands")

    hash_functions = generate_hash_functions(num_hashes, seed=seed)

    raw_text: list[str] = []
    grams_sets: list[set[str]] = []
    signatures: list[list[int]] = []

    # Pass 1: normalize, n-gram, minhash each document
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        raw_text.append(text)

        normalized_text = normalize_text(text)
        unique_grams = set(generate_ngrams(normalized_text, ngram_length))
        grams_sets.append(unique_grams)

        signatures.append(calculate_minhash(unique_grams, hash_functions))

    # Pass 2: LSH banding -- bucket documents by shared bands
    buckets: list[dict[tuple, list[int]]] = [defaultdict(list) for _ in range(num_bands)]
    for doc_idx, sig in enumerate(signatures):
        for band_idx, band in enumerate(band_signature(sig, num_bands)):
            buckets[band_idx][band].append(doc_idx)

    candidate_pairs: set[tuple[int, int]] = set()
    for bucket in buckets:
        for doc_list in bucket.values():
            for i in range(len(doc_list)):
                for j in range(i + 1, len(doc_list)):
                    candidate_pairs.add(tuple(sorted((doc_list[i], doc_list[j]))))

    # Pass 3: verify candidates with true Jaccard similarity
    uf = UnionFind(len(paths))
    for i, j in candidate_pairs:
        sim = jaccard_similarity(grams_sets[i], grams_sets[j])
        if sim >= jaccard_threshold:
            uf.union(i, j)

    # Pass 4: cluster and keep one survivor per cluster
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(paths)):
        clusters[uf.find(idx)].append(idx)

    to_remove: set[int] = set()
    rng = random.Random(seed)
    for members in clusters.values():
        if len(members) > 1:
            keep = rng.choice(members)
            to_remove.update(m for m in members if m != keep)

    # Write survivors
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, path in enumerate(paths):
        if idx not in to_remove:
            (output_dir / path.name).write_text(raw_text[idx], encoding="utf-8")

    return output_dir

def minhash_survivor_mask(
    texts: list[str],
    is_val: list[bool],
    num_hashes: int,
    num_bands: int,
    ngram_length: int,
    jaccard_threshold: float = DataConfig.deduplication_jaccard_threshold,
    seed: int = GlobalConfig.seed,
) -> list[bool]:
    """
    Compute a keep-mask for near-duplicate removal via MinHash + LSH, split-aware.

    When a cluster of near-duplicates spans both train and val, every train
    member is dropped and every val member is kept -- this is what prevents
    train/val leakage from near-duplicates. A train-only cluster keeps one
    random survivor.

    Args:
        texts (list[str]): Document texts, aligned with `is_val`.
        is_val (list[bool]): Whether each document belongs to val, aligned with `texts`.
        num_hashes (int): Number of hash functions in each MinHash signature.
        num_bands (int): Number of LSH bands (must evenly divide num_hashes).
        ngram_length (int): Word n-gram length used to represent each document.
        jaccard_threshold (float): Minimum true Jaccard similarity to confirm a duplicate.
        seed (int): Random seed for hash functions and train-only survivor choice.

    Returns:
        list[bool]: True for documents to keep, aligned with `texts`.
    """
    if num_hashes % num_bands != 0:
        raise ValueError("Number of hashes must be evenly divisible by the number of bands")

    hash_functions = generate_hash_functions(num_hashes, seed=seed)

    # Pass 1: signatures only. N-gram sets are NOT kept around for every
    # document -- at a few million docs that would hold 100+ GB of Python
    # sets in memory simultaneously. Signatures are cheap: one fixed-size
    # numpy row per doc.
    signatures = np.empty((len(texts), num_hashes), dtype=np.int64)
    start_time = time.monotonic()
    for idx, text in enumerate(texts):
        unique_grams = set(generate_ngrams(normalize_text(text), ngram_length))
        signatures[idx] = calculate_minhash(unique_grams, hash_functions)

        if (idx + 1) % 5000 == 0 or idx + 1 == len(texts):
            elapsed = time.monotonic() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            eta = (len(texts) - (idx + 1)) / rate if rate > 0 else None
            print(
                f"MinHash signatures | {idx + 1}/{len(texts)} docs | {rate:.0f} docs/s | "
                f"elapsed {format_duration(elapsed)} | "
                f"eta {format_duration(eta) if eta is not None else 'unknown'}"
            )

    candidate_pairs = find_candidate_pairs(signatures, num_bands)
    print(f"MinHash | {len(candidate_pairs)} candidate pairs to verify")

    # Pass 3: verify candidates with true Jaccard similarity. N-gram sets
    # are recomputed lazily, only for documents that actually landed in a
    # candidate pair (the large majority of documents never will), and
    # cached so a document appearing in several pairs isn't retokenized
    # repeatedly.
    @lru_cache(maxsize=10_000)
    def ngrams_for(idx: int) -> frozenset[str]:
        return frozenset(generate_ngrams(normalize_text(texts[idx]), ngram_length))

    uf = UnionFind(len(texts))
    verify_start = time.monotonic()
    for pair_idx, (i, j) in enumerate(candidate_pairs):
        sim = jaccard_similarity(ngrams_for(i), ngrams_for(j))
        if sim >= jaccard_threshold:
            uf.union(i, j)

        if (pair_idx + 1) % 5000 == 0 or pair_idx + 1 == len(candidate_pairs):
            print(
                f"MinHash verify | {pair_idx + 1}/{len(candidate_pairs)} pairs | "
                f"elapsed {format_duration(time.monotonic() - verify_start)}"
            )

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(texts)):
        clusters[uf.find(idx)].append(idx)

    keep = [True] * len(texts)
    rng = random.Random(seed)
    for members in clusters.values():
        if len(members) <= 1:
            continue

        val_members = [m for m in members if is_val[m]]
        if val_members:
            for m in members:
                if not is_val[m]:
                    keep[m] = False
        else:
            survivor = rng.choice(members)
            for m in members:
                if m != survivor:
                    keep[m] = False

    return keep

def minhash_dedup_streaming(
    val_input_dir: Path,
    train_input_dir: Path,
    val_output_dir: Path,
    train_output_dir: Path,
    num_hashes: int,
    num_bands: int,
    ngram_length: int,
    docs_per_chunk: int,
    jaccard_threshold: float = DataConfig.deduplication_jaccard_threshold,
    seed: int = GlobalConfig.seed,
) -> tuple[int, int]:
    """
    Remove near-duplicate documents via MinHash + LSH, streaming chunk_*.parquet
    files from disk to disk so memory stays bounded regardless of corpus size.

    Unlike `minhash_survivor_mask`, this never holds document text for the
    whole corpus in memory: only a compact per-doc signature (fixed-size
    numpy row) and a (chunk, row) locator are kept. During verification,
    text for the documents that land in a candidate pair is resolved in one
    grouped pass (one read per chunk actually needed), not lazily per pair.

    When a cluster of near-duplicates spans both train and val, every train
    member is dropped and every val member is kept -- this is what prevents
    train/val leakage from near-duplicates. A train-only cluster keeps one
    random survivor.

    Args:
        val_input_dir (Path): Directory of exact-deduped val chunk_*.parquet files.
        train_input_dir (Path): Directory of exact-deduped train chunk_*.parquet files.
        val_output_dir (Path): Directory to write final val chunks to.
        train_output_dir (Path): Directory to write final train chunks to.
        num_hashes (int): Number of hash functions in each MinHash signature.
        num_bands (int): Number of LSH bands (must evenly divide num_hashes).
        ngram_length (int): Word n-gram length used to represent each document.
        docs_per_chunk (int): Documents per output chunk file.
        jaccard_threshold (float): Minimum true Jaccard similarity to confirm a duplicate.
        seed (int): Random seed for hash functions and train-only survivor choice.

    Returns:
        tuple[int, int]: (final train count, final val count).
    """
    if num_hashes % num_bands != 0:
        raise ValueError("Number of hashes must be evenly divisible by the number of bands")

    chunk_infos: list[tuple[bool, Path, int]] = []
    for is_val, input_dir in ((True, val_input_dir), (False, train_input_dir)):
        for chunk_file in sorted(input_dir.glob("chunk_*.parquet")):
            num_rows = pq.ParquetFile(chunk_file).metadata.num_rows
            chunk_infos.append((is_val, chunk_file, num_rows))

    chunk_start_offsets: list[int] = []
    offset = 0
    for _, _, num_rows in chunk_infos:
        chunk_start_offsets.append(offset)
        offset += num_rows
    total_docs = offset

    hash_functions = generate_hash_functions(num_hashes, seed=seed)

    # Pass 1: signatures + a locator per doc (chunk index, row index) --
    # never the text itself. This is what keeps memory flat as corpus size grows.
    signatures = np.empty((total_docs, num_hashes), dtype=np.int64)
    doc_is_val = np.empty(total_docs, dtype=bool)
    doc_chunk_idx = np.empty(total_docs, dtype=np.int32)
    doc_row_idx = np.empty(total_docs, dtype=np.int32)

    global_idx = 0
    start_time = time.monotonic()
    for chunk_idx, (is_val, chunk_file, num_rows) in enumerate(chunk_infos):
        texts = pq.read_table(chunk_file, columns=["text"]).column("text").to_pylist()
        for row_idx, text in enumerate(texts):
            unique_grams = set(generate_ngrams(normalize_text(text), ngram_length))
            signatures[global_idx] = calculate_minhash(unique_grams, hash_functions)
            doc_is_val[global_idx] = is_val
            doc_chunk_idx[global_idx] = chunk_idx
            doc_row_idx[global_idx] = row_idx
            global_idx += 1

            if global_idx % 5000 == 0 or global_idx == total_docs:
                elapsed = time.monotonic() - start_time
                rate = global_idx / elapsed if elapsed > 0 else 0.0
                eta = (total_docs - global_idx) / rate if rate > 0 else None
                print(
                    f"MinHash signatures | {global_idx}/{total_docs} docs | {rate:.0f} docs/s | "
                    f"elapsed {format_duration(elapsed)} | "
                    f"eta {format_duration(eta) if eta is not None else 'unknown'}"
                )

    # Pass 2: LSH banding -- bucket documents by shared bands, one band at a
    # time (see find_candidate_pairs) so peak memory doesn't scale by num_bands.
    candidate_pairs = find_candidate_pairs(signatures, num_bands)
    print(f"MinHash | {len(candidate_pairs)} candidate pairs to verify")

    # Pass 3: verify candidates with true Jaccard similarity. Every doc
    # referenced by any candidate pair is resolved up front, grouped by
    # chunk, so each needed chunk is read exactly once -- doing this lazily
    # per pair (a per-doc LRU cache against 20k-row chunks) thrashed badly:
    # with candidate pairs scattered essentially randomly across chunks (LSH
    # bucketing has nothing to do with original file order), most lookups
    # were cache misses, each re-reading a full chunk for one row.
    verify_start = time.monotonic()
    needed_by_chunk: dict[int, set[int]] = defaultdict(set)
    for i, j in candidate_pairs:
        needed_by_chunk[int(doc_chunk_idx[i])].add(i)
        needed_by_chunk[int(doc_chunk_idx[j])].add(j)

    ngrams_by_idx: dict[int, frozenset[str]] = {}
    for chunk_idx, doc_indices in needed_by_chunk.items():
        _, chunk_file, _ = chunk_infos[chunk_idx]
        texts = pq.read_table(chunk_file, columns=["text"]).column("text").to_pylist()
        for idx in doc_indices:
            text = texts[int(doc_row_idx[idx])]
            ngrams_by_idx[idx] = frozenset(generate_ngrams(normalize_text(text), ngram_length))

    print(
        f"MinHash verify | resolved {len(ngrams_by_idx)} unique docs from "
        f"{len(needed_by_chunk)} chunks | elapsed {format_duration(time.monotonic() - verify_start)}"
    )

    uf = UnionFind(total_docs)
    for pair_idx, (i, j) in enumerate(candidate_pairs):
        sim = jaccard_similarity(ngrams_by_idx[i], ngrams_by_idx[j])
        if sim >= jaccard_threshold:
            uf.union(i, j)

        if (pair_idx + 1) % 5000 == 0 or pair_idx + 1 == len(candidate_pairs):
            print(
                f"MinHash verify | {pair_idx + 1}/{len(candidate_pairs)} pairs | "
                f"elapsed {format_duration(time.monotonic() - verify_start)}"
            )

    # Pass 4: cluster and resolve survivors, split-aware
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(total_docs):
        clusters[uf.find(idx)].append(idx)

    keep_mask = np.ones(total_docs, dtype=bool)
    rng = random.Random(seed)
    for members in clusters.values():
        if len(members) <= 1:
            continue

        val_members = [m for m in members if doc_is_val[m]]
        if val_members:
            for m in members:
                if not doc_is_val[m]:
                    keep_mask[m] = False
        else:
            survivor = rng.choice(members)
            for m in members:
                if m != survivor:
                    keep_mask[m] = False

    # Pass 5: stream through the input chunks one more time, writing only
    # surviving rows. Buffers hold at most `docs_per_chunk` documents.
    train_buffer: list[dict] = []
    val_buffer: list[dict] = []
    train_chunk_id = 0
    val_chunk_id = 0

    def flush(buffer: list[dict], output_dir: Path, chunk_id: int) -> int:
        if not buffer:
            return chunk_id
        batch = {col: [doc[col] for doc in buffer] for col in DataConfig.keep_columns}
        write_parquet(batch, output_dir / f"chunk_{chunk_id:05d}.parquet")
        buffer.clear()
        return chunk_id + 1

    for chunk_idx, (is_val, chunk_file, num_rows) in enumerate(chunk_infos):
        rows = pq.read_table(chunk_file).to_pylist()
        start = chunk_start_offsets[chunk_idx]
        buffer = val_buffer if is_val else train_buffer

        for row_idx, row in enumerate(rows):
            if keep_mask[start + row_idx]:
                buffer.append(row)
                if len(buffer) >= docs_per_chunk:
                    if is_val:
                        val_chunk_id = flush(val_buffer, val_output_dir, val_chunk_id)
                    else:
                        train_chunk_id = flush(train_buffer, train_output_dir, train_chunk_id)

    train_chunk_id = flush(train_buffer, train_output_dir, train_chunk_id)
    val_chunk_id = flush(val_buffer, val_output_dir, val_chunk_id)

    final_train_count = int((keep_mask & ~doc_is_val).sum())
    final_val_count = int((keep_mask & doc_is_val).sum())

    return final_train_count, final_val_count