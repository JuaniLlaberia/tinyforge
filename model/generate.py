import torch

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
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt).ids
    for _ in range(max_new_tokens):
        window = ids[-context_length:]
        inputs = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(inputs)
        next_id = int(logits[0, -1].argmax())
        ids.append(next_id)
        if next_id == eot_id:
            break
    return tokenizer.decode(ids)
