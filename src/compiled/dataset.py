"""
dataset.py — Generate addition problems as token sequences.

Used by both:
  - the compiled model (for evaluation / verification)
  - the trained model (for supervised training)

This is the SAME dataset for both, so the comparison is fair.
"""

import random
import torch
from torch.utils.data import Dataset
from .vocab import encode_addition, PAD, VOCAB_SIZE


class AdditionDataset(Dataset):
    """
    Each item is one addition problem: a + b = c.

    Returns tensors ready for a causal language model:
        tokens : [input... | SEP | c0 c1 ... EOS]   (full sequence)
        labels : [-100 ... -100 | c0 c1 ... EOS]    (-100 = ignore in loss)

    The model is trained (or evaluated) to predict each output digit
    given all previous tokens.
    """

    def __init__(
        self,
        n_digits: int,
        n_samples: int,
        seed: int = 42,
        max_val: int | None = None,
    ):
        """
        Args:
            n_digits  : number of digits in each operand (e.g. 3 → 0-999)
            n_samples : how many examples to generate
            seed      : random seed for reproducibility
            max_val   : override upper bound (default: 10**n_digits - 1)
        """
        self.n_digits  = n_digits
        self.n_samples = n_samples
        self.max_val   = max_val if max_val is not None else 10**n_digits - 1

        rng = random.Random(seed)
        self.examples = []
        for _ in range(n_samples):
            a = rng.randint(0, self.max_val)
            b = rng.randint(0, self.max_val)
            self.examples.append(encode_addition(a, b, n_digits))

        # All sequences have the same length for this n_digits
        #   input  : 2*(n_digits+1) + 1   (n_digits+1 pairs + SEP)
        #   output : (n_digits+1) + 1      (result digits + EOS)
        # n_pairs = n_digits + 1 (encode_addition uses an extra pair for final carry)
        self.n_pairs    = n_digits + 1
        self.input_len  = 2 * self.n_pairs + 1   # 2*n_digits + 3
        self.output_len = self.n_pairs + 1        # n_digits + 2
        self.seq_len    = self.input_len + self.output_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ex = self.examples[idx]
        full_seq = ex["input_tokens"] + ex["output_tokens"]

        tokens = torch.tensor(full_seq, dtype=torch.long)

        # Labels: -100 for input tokens (not predicted), digit tokens for output
        labels = torch.full_like(tokens, fill_value=-100)
        labels[self.input_len:] = tokens[self.input_len:]

        return {"tokens": tokens, "labels": labels, "meta": ex}


def make_split(
    n_digits_train: int,
    n_digits_test_list: list[int],
    n_train: int = 10_000,
    n_test:  int = 1_000,
    seed: int = 42,
) -> dict:
    """
    Build train set (n_digits_train) and multiple test sets
    at different digit lengths (for length generalization curves).

    The KEY experiment: train on n_digits_train, test on larger lengths.
    The compiled model ignores training — it works on all lengths.
    """
    return {
        "train": AdditionDataset(n_digits_train, n_train, seed=seed),
        "test":  {
            n: AdditionDataset(n, n_test, seed=seed + 1)
            for n in n_digits_test_list
        },
    }


if __name__ == "__main__":
    ds = AdditionDataset(n_digits=3, n_samples=5)
    for item in ds:
        meta = item["meta"]
        print(f"  {meta['a']} + {meta['b']} = {meta['result']}")
        print(f"  tokens : {item['tokens'].tolist()}")
        print(f"  labels : {item['labels'].tolist()}")
        print()
