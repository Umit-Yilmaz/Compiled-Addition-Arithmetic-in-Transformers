"""
metrics.py — Evaluation helpers for the addition experiments.

Provides:
  - eval_model_all_lengths   : run a model on multiple digit-length test sets
  - carry_chain_accuracy     : accuracy on adversarial carry-heavy inputs
  - per_position_accuracy    : digit-by-digit accuracy breakdown
"""

import torch
from ..compiled.vocab import encode_addition, decode_output, decode_output_token, EOS


def eval_model_all_lengths(
    model,
    test_lengths: list,
    n_test:       int  = 500,
    seed:         int  = 43,
) -> dict:
    """
    Evaluate model on AdditionDatasets for each length in test_lengths.

    Returns dict: {n_digits: accuracy}
    """
    from ..compiled.dataset import AdditionDataset
    results = {}
    for n in test_lengths:
        ds  = AdditionDataset(n, n_test, seed=seed)
        acc = _exact_match(model, ds)
        results[n] = round(acc, 4)
    return results


def _exact_match(model, dataset) -> float:
    correct = 0
    with torch.no_grad():
        for ex in dataset.examples:
            pred = model.generate(ex["a"], ex["b"], ex["n_digits"])
            if pred == ex["a"] + ex["b"]:
                correct += 1
    return correct / len(dataset)


def carry_chain_accuracy(model, max_digits: int = 10) -> dict:
    """
    Test accuracy on carry-chain inputs: (10^n - 1) + 1  (e.g. 999...9 + 1).
    These require propagating a carry across ALL digit positions.

    Returns dict: {n_digits: 1.0 or 0.0}
    """
    results = {}
    with torch.no_grad():
        for n in range(1, max_digits + 1):
            a = 10**n - 1   # e.g. 99 for n=2
            b = 1
            pred = model.generate(a, b, n)
            results[n] = 1.0 if pred == a + b else 0.0
    return results


def per_position_accuracy(model, n_digits: int, n_test: int = 500, seed: int = 43) -> list:
    """
    For each output digit position, compute fraction correct.
    Returns list of length n_digits+1 (position 0 = LSB, last = final carry).
    """
    from ..compiled.dataset import AdditionDataset
    ds = AdditionDataset(n_digits, n_test, seed=seed)
    n_pairs = n_digits + 1
    pos_correct = [0] * n_pairs
    pos_total   = [0] * n_pairs

    with torch.no_grad():
        for ex in ds.examples:
            a, b, nd = ex["a"], ex["b"], ex["n_digits"]
            encoded  = encode_addition(a, b, nd)
            target   = encoded["output_tokens"]   # carry-encoded, length n_pairs+1 (incl EOS)
            input_toks = encoded["input_tokens"]

            tokens = input_toks[:]
            for step in range(n_pairs):
                t    = torch.tensor(tokens, dtype=torch.long)
                pred_tok = model._forward_single(t)[-1].argmax().item() if hasattr(model, '_forward_single') \
                           else _predict_one(model, tokens, nd)
                tokens.append(pred_tok)

                # Compare predicted token to target (both carry-encoded)
                pred_digit, _  = decode_output_token(pred_tok)
                tgt_digit,  _  = decode_output_token(target[step])
                pos_total[step] += 1
                if pred_digit == tgt_digit:
                    pos_correct[step] += 1

    return [c / t if t > 0 else 0.0 for c, t in zip(pos_correct, pos_total)]


def _predict_one(model, tokens, n_digits):
    """Fallback: run forward and return argmax of last position."""
    t = torch.tensor(tokens, dtype=torch.long)
    with torch.no_grad():
        logits = model.forward(t, n_digits + 1) if hasattr(model, '_n_pairs_arg') \
                 else model(t)
    return logits[-1].argmax().item()
