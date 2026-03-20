"""
run_all.py — Run all experiment phases in order.

Usage:  python experiments/run_all.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib

PHASES = [
    ("Phase 1 — Compiled model evaluation",  "experiments.phase1_compiled"),
    ("Phase 3 — LLM baselines",               "experiments.phase3_llm"),
    ("Phase 4 — Mechanistic analysis",        "experiments.phase4_mechanistic"),
]

if __name__ == "__main__":
    for label, module_name in PHASES:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print('='*60)
        mod = importlib.import_module(module_name)
        mod.main()
    print("\nAll phases complete. See results/ directory.")
