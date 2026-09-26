import os
import time
from collections import Counter, defaultdict

import regex as re

from config import TokenizerConfig
from data.progress import format_duration
from tokenizer.helper import find_chunk_boundaries, save_vocab_and_merges

def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Function to train BPE tokenizer.

    Args:
        input_path (str): Path to the text file with the training data.
        vocab_size (int): A positive integer that defines the max final vocab.
        special_tokens (list[str]): A list of strings to add to the vocabulary.
    Returns:
        dict[int, bytes]: The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        list[tuple[bytes, bytes]]: A list of BPE merges produced from training.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    global_counts = Counter()

    # Step 1: Open raw corpus file in binary
    t0 = time.time()
    with open(input_path, "rb") as f:
        num_processes = os.cpu_count()
        # Step 2: Chunking by special token
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # Step 3: Pre-tokenization
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # 3.1 Remove special tokens
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            stories = re.split(re.escape("<|endoftext|>"), chunk)
            stories = [x for x in stories if x]
            # 3.2 GPT2 regex pre-tokens
            for story in stories:
                for match in re.finditer(PAT, story):
                    pretoken = match.group()
                    pretoken_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
                    global_counts[pretoken_bytes] += 1
    print(f"Finished corpus pre-processing in {time.time() - t0:.1f}s")

    merges: list[tuple[bytes, bytes]] = []
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for i, tok in enumerate(special_tokens):
        vocab[256 + i] = tok.encode("utf-8")

    # Reverse lookup so a merge whose resulting bytes already exist in vocab
    # (reachable via a different merge path elsewhere in the corpus, e.g.
    # "ing" formed once via "i"+"ng" and independently again via "in"+"g")
    # reuses that existing id instead of minting a duplicate one. Without
    # this, the vocab can end up with multiple ids mapping to identical
    # bytes and end up short of `vocab_size` unique, reachable tokens.
    bytes_to_existing_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
    next_id = 256 + len(special_tokens)

    pair_counts = Counter()
    pair_to_pretokens = defaultdict(set)
    for pretoken, freq in global_counts.items():
        for pair in zip(pretoken, pretoken[1:]):
            pair_counts[pair] += freq
            pair_to_pretokens[pair].add(pretoken)

    # Step 4: BPE training. Keeps merging until vocab actually reaches
    # vocab_size unique entries (not just vocab_size - 256 - len(special_tokens)
    # iterations), since a merge that reuses an existing id doesn't add a
    # new vocab slot and needs an extra iteration to compensate.
    t0 = time.time()
    target_vocab_size = vocab_size
    while len(vocab) < target_vocab_size:
        # Early stopping
        if not pair_counts:
            break

        pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        affected = list(pair_to_pretokens[pair])

        for old_pretoken in affected:
            freq = global_counts.pop(old_pretoken)

            for p in zip(old_pretoken, old_pretoken[1:]):
                pair_counts[p] -= freq
                pair_to_pretokens[p].discard(old_pretoken)

            main_list = list(old_pretoken)
            old_len = len(pair)
            new_token = pair[0] + pair[1]

            j = 0
            while j <= len(main_list) - old_len:
                if tuple(main_list[j:j + old_len]) == pair:
                    main_list[j:j + old_len] = [new_token]
                    j += 1
                else:
                    j += 1

            new_pretoken = tuple(main_list)
            global_counts[new_pretoken] = global_counts.get(new_pretoken, 0) + freq

            for p in zip(new_pretoken, new_pretoken[1:]):
                pair_counts[p] += freq
                pair_to_pretokens[p].add(new_pretoken)

        new_token = pair[0] + pair[1]
        merges.append(pair)

        if new_token not in bytes_to_existing_id:
            vocab[next_id] = new_token
            bytes_to_existing_id[new_token] = next_id
            next_id += 1

        if len(merges) % 500 == 0 or len(vocab) == target_vocab_size:
            elapsed = time.time() - t0
            remaining = target_vocab_size - len(vocab)
            rate = (len(vocab) - 256 - len(special_tokens)) / elapsed if elapsed > 0 else 0.0
            eta = remaining / rate if rate > 0 else None
            print(
                f"BPE training | {len(merges)} merges | vocab {len(vocab)}/{target_vocab_size} | "
                f"elapsed {format_duration(elapsed)} | "
                f"eta {format_duration(eta) if eta is not None else 'unknown'}"
            )

    print(f"Finished BPE training in {time.time() - t0:.1f}s")

    return vocab, merges

def main() -> None:
    vocab, merges = train_bpe(
        input_path=str(TokenizerConfig.sample_path),
        vocab_size=TokenizerConfig.vocab_size,
        special_tokens=TokenizerConfig.special_tokens,
    )
    save_vocab_and_merges(vocab, merges, TokenizerConfig.vocab_path, TokenizerConfig.merges_path)
    print(f"Saved {len(vocab)} vocab entries, {len(merges)} merges to "
          f"{TokenizerConfig.vocab_path} / {TokenizerConfig.merges_path}")

if __name__ == "__main__":
    main()