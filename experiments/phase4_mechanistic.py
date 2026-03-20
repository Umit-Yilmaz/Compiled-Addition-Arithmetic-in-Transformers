"""
phase4_mechanistic.py — Mechanistic analysis.

1. Attention maps: compiled model visualization
2. Carry probes: linear probe accuracy on residual stream
3. Adversarial carry-chain: 999...9 + 1

Saves results/phase4_*.json
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.compiled.model import CompiledAdditionTransformer
from src.analysis.adversarial  import run_adversarial_comparison
from src.analysis.probing      import run_probing_analysis
from src.analysis.attention_viz import run_attention_extraction


def main():
    # ── Load models ───────────────────────────────────────────────────────────
    compiled = CompiledAdditionTransformer()

    # ── 1. Adversarial carry-chain ─────────────────────────────────────────────
    print("\n─── Adversarial carry-chain ───")
    run_adversarial_comparison(compiled, max_digits=10, save_dir="results")

    # ── 2. Carry probes ────────────────────────────────────────────────────────
    print("\n─── Carry probes (linear probe on residual stream) ───")
    run_probing_analysis(
        compiled,
        n_digits_list=[1, 2, 3, 4, 5],
        n_samples=500,
        save_dir="results",
    )

    # ── 3. Attention extraction ────────────────────────────────────────────────
    print("\n─── Attention map extraction ───")
    run_attention_extraction(
        compiled,
        example=(347, 658, 3),
        save_dir="results",
    )

    print("\nPhase 4 complete. Results in results/phase4_*.json")


if __name__ == "__main__":
    main()
