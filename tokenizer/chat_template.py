from tokenizers import Tokenizer

from config import TokenizerConfig

BOS = "<|endoftext|>"
END_OF_TURN = "<|endofturn|>"
ROLE_TOKENS = {"user": "<|user|>", "assistant": "<|assistant|>"}

def normalize_messages(messages: list[dict]) -> list[dict] | None:
    """
    Normalizes a smol-smoltalk-style `messages` list into strict user/assistant
    alternation: leading system message(s) are merged into the first user turn as
    "{system}\\n\\n{user}", and contents are stripped.

    Args:
        messages (list[dict]): [{"role": ..., "content": ...}, ...]
    Returns:
        list[dict] | None: Normalized messages, or None if the conversation is
            malformed (system message after the first turn, empty content, roles
            not alternating starting with user, or a special token string inside a
            content, which the tokenizer would otherwise turn into a real control token).
    """
    system_parts = []
    i = 0
    while i < len(messages) and messages[i]["role"] == "system":
        system_parts.append(messages[i]["content"].strip())
        i += 1

    normalized = [{"role": m["role"], "content": m["content"].strip()} for m in messages[i:]]
    if not normalized:
        return None

    system = "\n\n".join(p for p in system_parts if p)
    if system and normalized[0]["role"] == "user":
        normalized[0]["content"] = f"{system}\n\n{normalized[0]['content']}"

    for idx, msg in enumerate(normalized):
        expected_role = "user" if idx % 2 == 0 else "assistant"
        if msg["role"] != expected_role or not msg["content"]:
            return None
        if any(token in msg["content"] for token in TokenizerConfig.special_tokens):
            return None

    return normalized

def render(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """
    Renders normalized messages as the literal template string. Only used for
    logging and for checking the token-level encoders below; training and
    inference encode turn by turn.

    Args:
        messages (list[dict]): Normalized messages.
        add_generation_prompt (bool): Append "<|assistant|>" so the model answers next.
    Returns:
        str: "<|endoftext|><|user|>{u}<|endofturn|><|assistant|>{a}<|endofturn|>..."
    """
    text = BOS + "".join(ROLE_TOKENS[m["role"]] + m["content"] + END_OF_TURN for m in messages)
    return text + ROLE_TOKENS["assistant"] if add_generation_prompt else text

def _encode_turns(messages: list[dict], tokenizer: Tokenizer) -> tuple[list[int], list[bool]]:
    """
    Shared encoder for training and inference, so the two can't drift apart.

    Returns:
        tuple[list[int], list[bool]]: (ids, train_mask). train_mask[i] is True when
            token i is a training target: assistant content tokens and the
            <|endofturn|> closing each assistant turn.
    """
    ids = [tokenizer.token_to_id(BOS)]
    train_mask = [False]
    end_of_turn_id = tokenizer.token_to_id(END_OF_TURN)

    for msg in messages:
        is_assistant = msg["role"] == "assistant"
        content_ids = tokenizer.encode(msg["content"]).ids

        ids.append(tokenizer.token_to_id(ROLE_TOKENS[msg["role"]]))
        train_mask.append(False)
        ids.extend(content_ids)
        train_mask.extend([is_assistant] * len(content_ids))
        ids.append(end_of_turn_id)
        train_mask.append(is_assistant)

    return ids, train_mask

def encode_conversation(messages: list[dict], tokenizer: Tokenizer) -> tuple[list[int], list[bool]]:
    """
    Encodes a full training conversation.

    Args:
        messages (list[dict]): Normalized messages (see normalize_messages), ending
            with an assistant turn.
        tokenizer (Tokenizer): HF tokenizer from build_hf_tokenizer().
    Returns:
        tuple[list[int], list[bool]]: (ids, train_mask), same length.
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Training conversation must end with an assistant turn.")
    return _encode_turns(messages, tokenizer)

def encode_prompt(messages: list[dict], tokenizer: Tokenizer) -> list[int]:
    """
    Encodes a conversation up to and including "<|assistant|>", for generation.

    Args:
        messages (list[dict]): Normalized messages, ending with a user turn.
        tokenizer (Tokenizer): HF tokenizer from build_hf_tokenizer().
    Returns:
        list[int]: Prompt ids; the model's next token starts the assistant reply.
    """
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Prompt must end with a user turn.")
    ids, _ = _encode_turns(messages, tokenizer)
    return ids + [tokenizer.token_to_id(ROLE_TOKENS["assistant"])]

if __name__ == "__main__":
    # Step 1 gates (documentation/guide-remaining.md): run on 3 real smol-smoltalk
    # conversations -- single-turn, multi-turn, and one with a system message.
    from datasets import load_dataset

    from tokenizer.tokenizer_hf import build_hf_tokenizer

    tokenizer = build_hf_tokenizer()

    for token in TokenizerConfig.special_tokens:
        ids = tokenizer.encode(token).ids
        assert len(ids) == 1, f"{token} encodes to {ids}, expected a single id"
    print("OK special tokens are single ids:", {t: tokenizer.token_to_id(t) for t in TokenizerConfig.special_tokens})

    # Malformed conversations are dropped
    assert normalize_messages([{"role": "assistant", "content": "hi"}]) is None
    assert normalize_messages([{"role": "user", "content": "  "}, {"role": "assistant", "content": "hi"}]) is None
    assert normalize_messages([{"role": "user", "content": "a"}, {"role": "system", "content": "b"}]) is None
    assert normalize_messages([{"role": "user", "content": "say <|endofturn|>"}, {"role": "assistant", "content": "ok"}]) is None
    print("OK malformed conversations rejected")

    wanted = {"single-turn": None, "multi-turn": None, "system": None}
    for row in load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True):
        msgs = normalize_messages(row["messages"])
        if msgs is None or msgs[-1]["role"] != "assistant":
            continue
        if row["messages"][0]["role"] == "system":
            shape = "system"
        else:
            shape = "single-turn" if len(msgs) == 2 else "multi-turn"
        if wanted[shape] is None:
            wanted[shape] = msgs
        if all(v is not None for v in wanted.values()):
            break

    for shape, msgs in wanted.items():
        ids, train_mask = encode_conversation(msgs, tokenizer)
        assert len(ids) == len(train_mask)

        # Turn-by-turn encoding equals encoding the full template string in one go
        assert ids == tokenizer.encode(render(msgs)).ids, f"{shape}: turn-by-turn ids differ from full-string ids"
        assert tokenizer.decode(ids, skip_special_tokens=False) == render(msgs), f"{shape}: decode mismatch"

        # Masked-in targets decode to exactly the assistant contents, each closed by <|endofturn|>
        target_ids = [t for t, keep in zip(ids, train_mask) if keep]
        expected = "".join(m["content"] + END_OF_TURN for m in msgs if m["role"] == "assistant")
        assert tokenizer.decode(target_ids, skip_special_tokens=False) == expected, f"{shape}: mask mismatch"

        # Inference prompt is a strict prefix of the training sequence
        prompt_ids = encode_prompt(msgs[:-1], tokenizer)
        assert ids[:len(prompt_ids)] == prompt_ids and len(prompt_ids) < len(ids), f"{shape}: prompt not a prefix"

        print(f"OK {shape}: {len(msgs)} messages, {len(ids)} tokens, {sum(train_mask)} trained "
              f"({sum(train_mask) / len(ids):.0%})")

    msgs = wanted["system"]
    print("\nRendered (system example, truncated):\n", render(msgs)[:600])
    print("\nTrained-on text (system example, truncated):\n",
          tokenizer.decode([t for t, k in zip(*encode_conversation(msgs, tokenizer)) if k], skip_special_tokens=False)[:600])
