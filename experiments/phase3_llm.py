"""
phase3_llm.py — LLM baseline evaluation.

Uses mock results by default (no API key needed).
Set ANTHROPIC_API_KEY or OPENAI_API_KEY to use real models.

Saves results/phase3_llm.json
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.run_baselines import eval_llm_baseline


def main():
    # Determine backend
    backend = "mock"
    if os.environ.get("ANTHROPIC_API_KEY"):
        backend = "claude"
    elif os.environ.get("OPENAI_API_KEY"):
        backend = "openai"

    print(f"Backend: {backend}")
    print("Evaluating LLM baselines...")

    results = eval_llm_baseline(
        test_lengths = list(range(1, 11)),
        n_test       = 200,
        backend      = backend,
        save_dir     = "results",
    )

    print("\nSummary:")
    print(f"  {'n':>4}  {'zero-shot':>10}  {'CoT':>8}  {'tool':>8}")
    for n in range(1, 11):
        z = results["results"]["zero_shot"][str(n)]
        c = results["results"]["cot"][str(n)]
        t = results["results"]["tool"][str(n)]
        print(f"  {n:4d}  {z:10.4f}  {c:8.4f}  {t:8.4f}")


if __name__ == "__main__":
    main()
