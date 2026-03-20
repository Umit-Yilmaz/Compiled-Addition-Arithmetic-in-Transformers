"""
phase1_compiled.py — Evaluate the compiled model on all digit lengths.

Saves results/phase1_compiled.json
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
from src.compiled.model import CompiledAdditionTransformer
from src.eval.metrics import eval_model_all_lengths, carry_chain_accuracy


def main():
    os.makedirs("results", exist_ok=True)
    model = CompiledAdditionTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Compiled model — {n_params:,} parameters (all frozen)")

    # N_MAX=10 supports n_pairs up to 10, i.e. n_digits up to 9.
    # (n_digits=10 -> n_pairs=11 > N_MAX, step one-hot would exceed embedding dim.)
    test_lengths = list(range(1, 10))
    print("\nEvaluating across digit lengths 1-9 (N_MAX=10 -> max n_digits=9)...")

    acc = eval_model_all_lengths(model, test_lengths, n_test=500)
    for n, a in acc.items():
        print(f"  n={n:2d}: {a:.4f}")

    print("\nCarry-chain adversarial tests...")
    carry = carry_chain_accuracy(model, max_digits=9)
    for n, a in carry.items():
        status = "OK" if a == 1.0 else "FAIL"
        print(f"  {10**n-1} + 1 = {10**n}  [{status}]")

    results = {
        "model": "compiled",
        "n_params": n_params,
        "accuracy_by_length": {str(k): v for k, v in acc.items()},
        "carry_chain":        {str(k): v for k, v in carry.items()},
    }
    with open("results/phase1_compiled.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → results/phase1_compiled.json")


if __name__ == "__main__":
    main()
