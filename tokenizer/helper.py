import json
import os
from pathlib import Path
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.

    Args:
        file (BinaryIO): Binary file handle to scan for chunk boundaries.
        desired_num_chunks (int): Target number of chunks to split the file into.
        split_special_token (bytes): Token boundaries are snapped to, so chunks never split it.
    Returns:
        list[int]: Sorted, deduplicated byte offsets delimiting each chunk.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def calculate_compression(num_characters: int, num_bpe_tokens: int) -> float:
    """
    Calculates the compression ratio of BPE tokenization.

    Args:
        num_characters (int): Total count of characters (or bytes) in raw text.
        num_bpe_tokens (int): Total count of tokens produced after BPE encoding.
    Returns:
        float: The compression ratio (e.g., 4.2 means 4.2 characters per token).
    """
    if num_bpe_tokens <= 0:
        raise ValueError("Number of tokens must be greater than zero.")

    return num_characters / num_bpe_tokens

def bytes_to_unicode() -> dict[int, str]:
    """
    Maps every byte value (0-255) to a unique printable unicode character,
    so arbitrary byte tokens can round-trip safely through JSON string keys
    and merges.txt text lines. The standard GPT-2 byte-level BPE trick, and
    what makes vocab.json/merges.txt loadable by HF `tokenizers` directly.

    Returns:
        dict[int, str]: Mapping from byte value to its unicode representation.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))

def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: Path,
    merges_path: Path,
) -> None:
    """
    Serialize a trained BPE vocabulary and merge list to HF `tokenizers`'
    native BPE format: vocab.json (token string -> id) and merges.txt (one
    "tokenA tokenB" pair per line), using the GPT-2 byte-to-unicode mapping
    so arbitrary byte tokens round-trip safely through both formats.

    Args:
        vocab (dict[int, bytes]): Mapping from token id to raw token bytes.
        merges (list[tuple[bytes, bytes]]): Ordered BPE merge rules.
        vocab_path (Path): Where to write vocab.json.
        merges_path (Path): Where to write merges.txt.
    Returns:
        None.
    """
    byte_encoder = bytes_to_unicode()

    def encode_bytes(token_bytes: bytes) -> str:
        return "".join(byte_encoder[b] for b in token_bytes)

    vocab_json = {encode_bytes(token_bytes): token_id for token_id, token_bytes in vocab.items()}

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False)

    merges_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for left, right in merges:
            f.write(f"{encode_bytes(left)} {encode_bytes(right)}\n")

def load_vocab_and_merges(vocab_path: Path, merges_path: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Load a BPE vocabulary and merge list from HF `tokenizers`' native BPE
    format, the inverse of `save_vocab_and_merges`.

    Args:
        vocab_path (Path): Path to a vocab.json file (token string -> id).
        merges_path (Path): Path to a merges.txt file.
    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: (vocab, merges).
    """
    byte_decoder = {unicode_char: b for b, unicode_char in bytes_to_unicode().items()}

    def decode_str(token_str: str) -> bytes:
        return bytes(byte_decoder[c] for c in token_str)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab = {token_id: decode_str(token_str) for token_str, token_id in vocab_json.items()}

    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            left, right = line.split(" ")
            merges.append((decode_str(left), decode_str(right)))

    return vocab, merges