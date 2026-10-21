"""
Step 3 gates that don't need a training run (documentation/guide-remaining.md).
The overfit and 20-step gates are train/sft.py runs; see the guide for the commands.
"""
import json
from collections import Counter

import torch
import torch.nn.functional as F

from config import GlobalConfig, SFTConfig, TokenizerConfig
from model.helper.sft_loader import SFTDataset, resolve_sft_file
from tokenizer.chat_template import END_OF_TURN
from tokenizer.tokenizer_hf import build_hf_tokenizer
from train.sft import build_model, count_targets, init_chat_token_embeddings, masked_loss_sum

def check_epoch_plan(dataset: SFTDataset, micro_batch_size: int, grad_accum_steps: int) -> None:
    plans = [dataset.epoch_steps(epoch, micro_batch_size, grad_accum_steps) for epoch in (0, 1)]
    for epoch, steps in enumerate(plans):
        seen = Counter(i for step in steps for micro in step for i in micro)
        assert sorted(seen) == list(range(len(dataset))) and set(seen.values()) == {1}, \
            f"epoch {epoch}: not every conversation exactly once"
        assert all(len(step) <= grad_accum_steps and all(len(m) <= micro_batch_size for m in step) for step in steps)
    assert plans[0] == dataset.epoch_steps(0, micro_batch_size, grad_accum_steps), "epoch plan not reproducible"
    assert plans[0] != plans[1], "epochs 0 and 1 have the same order"
    assert len(plans[0]) == len(plans[1]), "steps per epoch must not depend on the epoch"

    padded = sum(max(dataset.lengths[i] for i in m) * len(m) for step in plans[0] for m in step)
    print(f"OK epoch plan: {len(plans[0])} steps/epoch, each conversation once, reproducible, "
          f"padding overhead {padded / sum(dataset.lengths) - 1:.1%}")

def check_collate(dataset: SFTDataset, tokenizer, raw_messages: list[list[dict]]) -> None:
    indices = list(range(8))
    inputs, labels = dataset.collate(indices)
    assert inputs.shape == labels.shape == (8, max(dataset.lengths[i] for i in indices) - 1)

    for row, i in enumerate(indices):
        n = dataset.lengths[i] - 1
        assert inputs[row, :n].tolist() == dataset.ids[i][:-1].tolist()
        assert (inputs[row, n:] == TokenizerConfig.eot_id).all() and (labels[row, n:] == SFTConfig.ignore_index).all()

        targets = labels[row][labels[row] != SFTConfig.ignore_index].tolist()
        expected = "".join(m["content"] + END_OF_TURN for m in raw_messages[i] if m["role"] == "assistant")
        assert tokenizer.decode(targets, skip_special_tokens=False) == expected, f"row {row}: labels != assistant text"
    print("OK collate: padding, shift and labels decode to exactly the assistant turns")

def check_loss_normalization() -> None:
    torch.manual_seed(0)
    vocab = TokenizerConfig.vocab_size

    # One target token: masked sum equals that token's cross-entropy
    logits = torch.randn(2, 5, vocab)
    labels = torch.full((2, 5), SFTConfig.ignore_index)
    labels[1, 3] = 42
    assert torch.allclose(masked_loss_sum(logits, labels) / count_targets(labels), F.cross_entropy(logits[1, 3], torch.tensor(42)))

    # Two micro-batches with different target counts: sum / step total == mean over all targets pooled
    micro = [(torch.randn(2, 6, vocab), torch.randint(0, vocab, (2, 6))) for _ in range(2)]
    micro[0][1][:, :5] = SFTConfig.ignore_index   # 2 targets
    micro[1][1][0, :2] = SFTConfig.ignore_index   # 10 targets
    step_targets = sum(count_targets(l) for _, l in micro)
    step_loss = sum(masked_loss_sum(lg, l) for lg, l in micro) / step_targets
    pooled_logits = torch.cat([lg.reshape(-1, vocab) for lg, _ in micro])
    pooled_labels = torch.cat([l.reshape(-1) for _, l in micro])
    assert torch.allclose(step_loss, F.cross_entropy(pooled_logits, pooled_labels, ignore_index=SFTConfig.ignore_index))
    print("OK loss normalization: every target token weighs the same across micro-batches")

def check_chat_token_init() -> None:
    model = build_model("xs", torch.device("cpu"))
    weight = model.embedding_matrix.embedding_matrix
    eot_before = weight[TokenizerConfig.eot_id].clone()
    init_chat_token_embeddings(model)

    regular = torch.ones(weight.size(0), dtype=torch.bool)
    regular[256:256 + len(TokenizerConfig.special_tokens)] = False
    mean_row = weight[regular].mean(dim=0)
    for token_id in (257, 258, 259):
        assert torch.equal(weight[token_id], mean_row)
    assert torch.equal(weight[TokenizerConfig.eot_id], eot_before), "<|endoftext|> row must be untouched"
    assert model.head.W is weight, "head must still be tied to the embedding"
    print("OK chat-token init: rows 257-259 = mean regular row, EOT untouched, head still tied")

def main() -> None:
    tokenizer = build_hf_tokenizer()
    path = resolve_sft_file(SFTConfig.train_ablation_file)
    dataset = SFTDataset(path, tokenizer, GlobalConfig.seed)
    raw_messages = [json.loads(line)["messages"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    check_epoch_plan(dataset, SFTConfig.micro_batch_size, SFTConfig.grad_accum_steps)
    check_collate(dataset, tokenizer, raw_messages)
    check_loss_normalization()
    check_chat_token_init()

if __name__ == "__main__":
    main()
