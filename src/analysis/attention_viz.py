"""
attention_viz.py — Extract and visualize attention patterns.

Extracts the (seq_len, seq_len) attention weight matrices for all 3 heads
from the compiled model, on a given example, and saves them to JSON
for plotting in visualize.py.
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from ..compiled.vocab import encode_addition


def _extract_attention_compiled(model, a: int, b: int, n_digits: int) -> dict:
    """
    Return attention weights for all 3 heads.
    Returns dict: {head_idx: np.ndarray (seq_len, seq_len)}
    """
    enc        = encode_addition(a, b, n_digits)
    input_toks = enc["input_tokens"]
    output_toks = enc["output_tokens"][:-1]   # drop EOS
    n_pairs    = enc["n_pairs"]

    full_seq = input_toks + output_toks
    t = torch.tensor(full_seq, dtype=torch.long)
    T = len(full_seq)

    with torch.no_grad():
        x = model.embedding(t) + model._positional_encoding(t, n_pairs)
        attn_mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)

        heads = {}
        for h in range(3):
            Q = x @ model.W_Q[h]
            K = x @ model.W_K[h]
            scores  = Q @ K.T + attn_mask
            weights = F.softmax(scores, dim=-1)
            heads[h] = weights.numpy()

    return heads


def _make_token_labels(a: int, b: int, n_digits: int) -> list:
    """Human-readable token labels for the sequence."""
    enc = encode_addition(a, b, n_digits)
    from ..compiled.vocab import TOKEN_TO_STR
    labels = [TOKEN_TO_STR[t] for t in enc["input_tokens"]]
    labels += [TOKEN_TO_STR[t] for t in enc["output_tokens"][:-1]]
    return labels


def run_attention_extraction(
    compiled_model,
    example: tuple      = (347, 658, 3),
    save_dir: str       = "results",
) -> dict:
    """
    Extract attention matrices for the compiled model on a given example.
    Saves to JSON-serializable numpy arrays.
    """
    os.makedirs(save_dir, exist_ok=True)
    a, b, n_digits = example

    comp_heads = _extract_attention_compiled(compiled_model, a, b, n_digits)
    labels     = _make_token_labels(a, b, n_digits)

    out = {
        "example":      {"a": a, "b": b, "n_digits": n_digits, "result": a + b},
        "token_labels": labels,
        "compiled": {str(h): comp_heads[h].tolist() for h in range(3)},
    }

    out_path = os.path.join(save_dir, "phase4_attention.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {out_path}")
    return out
