"""
adversarial.py — Carry-chain stress tests.

The hardest inputs for addition are those requiring a carry to propagate
across all digit positions:  (10^n - 1) + 1  →  10^n

These test whether the carry mechanism is truly general.
"""

import json
import os
import random
import torch


def carry_chain_suite(model, max_digits: int = 10) -> dict:
    """
    Test each model on the adversarial carry-chain inputs.

    For n digits: a = 10^n - 1  (e.g. 999 for n=3), b = 1
    Expected result: 10^n  (full carry propagation)

    Returns dict: {n_digits: {"a": int, "b": int, "expected": int, "predicted": int, "correct": bool}}
    """
    results = {}
    with torch.no_grad():
        for n in range(1, max_digits + 1):
            a = 10**n - 1
            b = 1
            expected = a + b
            predicted = model.generate(a, b, n)
            results[n] = {
                "a": a, "b": b,
                "expected":  expected,
                "predicted": predicted,
                "correct":   predicted == expected,
            }
    return results


def random_long_suite(model, n_digits: int, n_test: int = 200, seed: int = 99) -> dict:
    """
    Test accuracy on random n-digit problems (extension beyond training).
    """
    rng = random.Random(seed)
    max_val = 10**n_digits - 1
    correct = 0
    with torch.no_grad():
        for _ in range(n_test):
            a = rng.randint(0, max_val)
            b = rng.randint(0, max_val)
            pred = model.generate(a, b, n_digits)
            if pred == a + b:
                correct += 1
    return {"n_digits": n_digits, "n_test": n_test, "accuracy": correct / n_test}


def run_adversarial_comparison(
    compiled_model,
    max_digits: int = 10,
    save_dir:   str = "results",
) -> dict:
    """
    Run carry-chain tests on the compiled model, save JSON.
    """
    os.makedirs(save_dir, exist_ok=True)

    compiled_carry = carry_chain_suite(compiled_model, max_digits)
    compiled_carry_acc = {n: int(v["correct"]) for n, v in compiled_carry.items()}

    out = {
        "carry_chain": {
            "compiled": compiled_carry_acc,
        },
        "carry_chain_details": {
            "compiled": {str(k): v for k, v in compiled_carry.items()},
        },
    }

    out_path = os.path.join(save_dir, "phase4_adversarial.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Print summary
    print("Carry-chain accuracy:")
    print(f"  {'n':>4}  {'compiled':>10}")
    for n in range(1, max_digits + 1):
        c = "✓" if compiled_carry_acc[n] else "✗"
        print(f"  {n:4d}  {c:>10}")

    return out
