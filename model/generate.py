import torch

@torch.no_grad()
def greedy_generate_ids(model, prompt_ids: list[int], context_length: int, stop_ids: set[int], max_new_tokens: int = 40) -> list[int]:
    """
    Naive greedy decode over token ids, no KV cache. Feeds the last `context_length`
    tokens back in on every step.

    Args:
        model: TransformerLM in eval mode.
        prompt_ids (list[int]): Token ids to condition on.
        context_length (int): Model's max input window.
        stop_ids (set[int]): Token ids that stop generation (the stop token is kept
            in the output).
        max_new_tokens (int): Maximum number of tokens to generate.
    Returns:
        list[int]: Only the newly generated ids.
    """
    device = next(model.parameters()).device
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        window = ids[-context_length:]
        inputs = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(inputs)
        next_id = int(logits[0, -1].argmax())
        ids.append(next_id)
        if next_id in stop_ids:
            break
    return ids[len(prompt_ids):]

@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, context_length: int, eot_id: int, max_new_tokens: int = 40) -> str:
    """
    Naive greedy decode, no KV cache . Feeds the last `context_length` tokens back in on every step.

    Args:
        model: TransformerLM in eval mode.
        tokenizer: HF `tokenizers.Tokenizer` with .encode()/.decode().
        prompt (str): Text to condition on.
        context_length (int): Model's max input window.
        eot_id (int): Token id that stops generation early.
        max_new_tokens (int): Maximum number of tokens to generate.
    Returns:
        str: Decoded prompt + completion.
    """
    prompt_ids = tokenizer.encode(prompt).ids
    new_ids = greedy_generate_ids(model, prompt_ids, context_length, {eot_id}, max_new_tokens)
    return tokenizer.decode(prompt_ids + new_ids)
