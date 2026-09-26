from pathlib import Path
from typing import Iterable, Iterator

import regex as re

from tokenizer.helper import load_vocab_and_merges

class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        """
        Initializes the BPE tokenizer from a vocabulary and an ordered merge list.

        Args:
            vocab (dict[int, bytes]): Mapping from token ID to raw token bytes.
            merges (list[tuple[bytes, bytes]]): Ordered list of BPE merge rules.
            special_tokens (list[str] | None): Special tokens to treat as atomic units.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.bytes_to_id = {b: idx for idx, b in vocab.items()}
        self.special_token_to_id = {
            tok: self.bytes_to_id[tok.encode("utf-8")]
            for tok in self.special_tokens}

        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(t) for t in sorted_tokens)
            self.special_re = re.compile(f"({pattern})")
            self.max_special_len = max(len(t) for t in self.special_tokens)
        else:
            self.special_re = None
            self.max_special_len = 0

    @classmethod
    def from_files(cls,
                    vocab_filepath: str,
                    merges_filepath: str,
                    special_tokens: list[str] | None = None) -> "Tokenizer":
        """
        Builds a Tokenizer by loading vocab.json/merges.txt from disk -- the
        same HF `tokenizers`-format files the fast encoder in Step 6 loads,
        so both encoders are guaranteed to read an identical vocabulary.

        Args:
            vocab_filepath (str): Path to a vocab.json file (token string -> id).
            merges_filepath (str): Path to a merges.txt file.
            special_tokens (list[str] | None): Special tokens to treat as atomic units.
        Returns:
            Tokenizer: A tokenizer built from the loaded vocabulary and merges.
        """
        vocab, merges = load_vocab_and_merges(Path(vocab_filepath), Path(merges_filepath))
        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, pretoken_bytes: tuple[bytes, ...]) -> tuple[bytes, ...]:
        """
        Repeatedly merges the highest-priority adjacent byte pair until none remain.

        Args:
            pretoken_bytes (tuple[bytes, ...]): Pretoken represented as single-byte tokens.
        Returns:
            tuple[bytes, ...]: Pretoken bytes after applying all eligible BPE merges.
        """
        parts = list(pretoken_bytes)
        
        while True:
            candidates = [p for p in zip(parts, parts[1:]) if p in self.merge_ranks]
            if not candidates:
                break

            pair = min(candidates, key=lambda p: self.merge_ranks[p])
            old_len = len(pair)
            new_token = pair[0] + pair[1]
            j = 0

            while j <= len(parts) - old_len:
                if tuple(parts[j:j + old_len]) == pair:
                    parts[j:j + old_len] = [new_token]
                    j += 1
                else:
                    j += 1

        return tuple(parts)

    def _pretoken_to_ids(self, pretoken: str) -> list[int]:
        """
        Converts a single pretoken string into its final token IDs after BPE merging.

        Args:
            pretoken (str): A pretoken produced by the pretokenization regex.
        Returns:
            list[int]: Token IDs for the merged pretoken.
        """
        pretoken_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
        merged = self._apply_merges(pretoken_bytes)
        
        return [self.bytes_to_id[b] for b in merged]

    def _emit_piece_ids(self, piece: str) -> Iterator[int]:
        """
        Pretokenizes a non-special text piece and yields token IDs for each pretoken.

        Args:
            piece (str): Plain text piece containing no special tokens.
        Returns:
            Iterator[int]: Token IDs for the piece, in order.
        """
        for match in self.PAT.finditer(piece):
            yield from self._pretoken_to_ids(match.group())

    def encode(self, text: str) -> list[int]:
        """
        Encodes a full text string into a list of token IDs.

        Args:
            text (str): Input text to tokenize.
        Returns:
            list[int]: Token IDs representing the text.
        """
        ids = []

        pieces = self.special_re.split(text) if self.special_re is not None else [text]
        for piece in pieces:
            if piece in self.special_token_to_id:
                ids.append(self.special_token_to_id[piece])
            elif piece:
                ids.extend(self._emit_piece_ids(piece))

        return ids

    def _drain(self, buf: str) -> Iterator[int]:
        """
        Encodes all complete pretokens in a streaming buffer, yielding their IDs.

        Args:
            buf (str): Accumulated text buffer that may end mid-pretoken.
        Returns:
            Iterator[int]: Token IDs for all pretokens that are safely complete.
            The unconsumed tail of the buffer is returned via StopIteration.value.
        """
        pieces = self.special_re.split(buf) if self.special_re is not None else [buf]

        for piece in pieces[:-1]:
            if piece in self.special_token_to_id:
                yield self.special_token_to_id[piece]
            elif piece:
                yield from self._emit_piece_ids(piece)

        tail = pieces[-1]

        margin = max(0, self.max_special_len - 1)
        safe_len = len(tail) - margin
        if safe_len <= 0:
            return tail

        safe, risky = tail[:safe_len], tail[safe_len:]

        matches = list(self.PAT.finditer(safe))
        if len(matches) <= 1:
            return safe + risky

        cut = matches[-1].start()
        for match in matches[:-1]:
            yield from self._pretoken_to_ids(match.group())
        return safe[cut:] + risky

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encodes an iterable of text chunks (e.g. a file handle) into token IDs.

        Args:
            iterable (Iterable[str]): Sequence of text chunks to encode in order.
        Returns:
            Iterator[int]: Token IDs streamed as chunks become available.
        """
        buf = ""
        MIN_BLOCK = 1 << 20 

        for chunk in iterable:
            if not chunk:
                continue
            buf += chunk
            if len(buf) < MIN_BLOCK:
                continue
            buf = yield from self._drain(buf)

        if buf:
            for _id in self.encode(buf):
                yield _id

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into a text string.

        Args:
            ids (list[int]): Token IDs to decode.
        Returns:
            str: The decoded text.
        """
        text = b"".join(self.vocab[id] for id in ids)
        text = text.decode("utf-8", errors="replace")
        return text