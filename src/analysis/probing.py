"""
probing.py — Linear probes for carry state in the residual stream.

For each layer / position, train a logistic regression to predict the
carry_in value from the residual activations.  High probe accuracy at
the output positions means the model has a linear carry representation.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ..compiled.vocab import encode_addition, decode_output_token, EOS
from ..compiled.dataset import AdditionDataset


# ── Activation extraction ─────────────────────────────────────────────────────

def _get_residuals_compiled(model, examples: list) -> tuple:
    """
    Extract residual stream at output positions for the compiled model.
    Returns (activations, carry_labels) as numpy arrays.
    """
    activations = []
    carry_labels = []

    with torch.no_grad():
        for ex in examples:
            a, b, n_digits = ex["a"], ex["b"], ex["n_digits"]
            enc = encode_addition(a, b, n_digits)
            input_toks = enc["input_tokens"]
            output_toks = enc["output_tokens"]   # carry-encoded
            n_pairs = enc["n_pairs"]
            input_len = len(input_toks)

            full_seq = input_toks + output_toks[:-1]   # excl EOS for forward pass
            t = torch.tensor(full_seq, dtype=torch.long)

            # Embed + positional encoding  (same as model.forward but intercept residual)
            x = model.embedding(t) + model._positional_encoding(t, n_pairs)
            attn_mask = torch.triu(torch.full((len(full_seq), len(full_seq)), float('-inf')), diagonal=1)

            attn_out = torch.zeros_like(x)
            for h in range(3):
                Q = x @ model.W_Q[h]
                K = x @ model.W_K[h]
                V = x @ model.W_V[h]
                scores  = Q @ K.T + attn_mask
                weights = torch.softmax(scores, dim=-1)
                attn_out += (weights @ V) @ model.W_O[h]
            x_post_attn = x + attn_out   # residual after attention, before MLP

            # Extract residual at each output position
            for step in range(n_pairs):
                pos = input_len + step
                if pos >= len(full_seq):
                    break
                # carry_in at this step = carry_out of previous token
                if step == 0:
                    carry_in = 0
                else:
                    _, c_out = decode_output_token(output_toks[step - 1])
                    carry_in = c_out

                activations.append(x_post_attn[pos].numpy())
                carry_labels.append(carry_in)

    return np.array(activations), np.array(carry_labels)


# ── Probe training ────────────────────────────────────────────────────────────

def train_carry_probe(activations: np.ndarray, labels: np.ndarray) -> dict:
    """
    Train a logistic regression probe to predict carry_in from residual activations.
    Returns dict with train_acc and test_acc.
    """
    if len(np.unique(labels)) < 2:
        return {"train_acc": 1.0, "test_acc": 1.0, "note": "only one class"}

    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc  = accuracy_score(y_test,  clf.predict(X_test))
    return {"train_acc": round(train_acc, 4), "test_acc": round(test_acc, 4)}


def run_probing_analysis(
    compiled_model,
    n_digits_list: list = None,
    n_samples:     int  = 500,
    save_dir:      str  = "results",
) -> dict:
    """Run carry probes on the compiled model and save results."""
    os.makedirs(save_dir, exist_ok=True)
    if n_digits_list is None:
        n_digits_list = [1, 2, 3, 4, 5]

    results = {"compiled": {}}

    for n in n_digits_list:
        ds = AdditionDataset(n, n_samples, seed=77)

        acts_c, labs_c = _get_residuals_compiled(compiled_model, ds.examples)
        probe_c = train_carry_probe(acts_c, labs_c)
        results["compiled"][n] = probe_c

        print(f"  n={n}  compiled={probe_c['test_acc']:.3f}")

    out = {"carry_probe": {k: {str(n): v for n, v in d.items()} for k, d in results.items()}}
    out_path = os.path.join(save_dir, "phase4_probing.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {out_path}")
    return out
