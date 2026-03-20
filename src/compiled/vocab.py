"""
vocab.py — Token vocabulary and input/output format.

════════════════════════════════════════════════════════════════
KEY DESIGN DECISION: carry is encoded in the output token ID
════════════════════════════════════════════════════════════════

In a 1-layer transformer, attention reads the PRE-MLP residual.
This means head 2 (carry) at position c_i would read c_{i-1}'s
carry BEFORE c_{i-1}'s MLP has computed it — always getting 0.

Fix: store carry in the TOKEN ID itself.
  - Output token  d   (d in 0..9) → digit=d, carry_out=0
  - Output token 10+d (d in 0..9) → digit=d, carry_out=1

The carry is now in the EMBEDDING (accessible to attention immediately),
not in the residual (which requires MLP to be computed first).
This makes 1 layer sufficient for exact carry propagation.

════════════════════════════════════════════════════════════════
INPUT FORMAT
════════════════════════════════════════════════════════════════

INPUT  (condition):  [a₀ b₀  a₁ b₁  …  aₙ bₙ  SEP]
OUTPUT (generated):  [out₀  out₁  …  outₙ  EOS]

  aᵢ   = (A // 10ⁱ) % 10     digit of A at position i (LSB first)
  outᵢ = digit_i + 10 * carry_out_i

  At step i:
    digit_i    = (aᵢ + bᵢ + carry_in_i) % 10
    carry_out_i = (aᵢ + bᵢ + carry_in_i) // 10
    carry_in_0  = 0  (always)
    carry_in_i  = carry_out_{i-1}

We use n+1 input pairs (one extra pair of zeros) so that every output step
i ∈ {0, …, n} has a matching input pair — including the final carry step.
"""

# ── Token IDs ───────────────────────────────────────────────────────────────

# INPUT operand digits:  tokens 0-9  (same as carry=0 output digits)
DIGITS_IN  = list(range(10))

# OUTPUT carry=0 tokens: tokens 0-9   (digit = token id)
DIGITS_OUT0 = list(range(10))

# OUTPUT carry=1 tokens: tokens 10-19  (digit = token - 10)
DIGITS_OUT1 = list(range(10, 20))

SEP      = 20    # separates input pairs from output region
EOS      = 21    # end-of-sequence
PAD      = 22    # padding

VOCAB_SIZE = 23  # 0-9 (input/out-c0), 10-19 (out-c1), 20 SEP, 21 EOS, 22 PAD

TOKEN_TO_STR = {
    **{i:      str(i)        for i in range(10)},    # 0-9
    **{10 + i: f"{i}'"       for i in range(10)},    # 10-19  (digit with carry)
    SEP: "|",
    EOS: "<eos>",
    PAD: "<pad>",
}
STR_TO_TOKEN = {v: k for k, v in TOKEN_TO_STR.items()}


def output_token(digit: int, carry_out: int) -> int:
    """Encode digit + carry_out into a single output token id."""
    return digit + 10 * carry_out


def decode_output_token(tok: int) -> tuple[int, int]:
    """Decode output token → (digit, carry_out)."""
    if tok < 10:
        return tok, 0
    elif tok < 20:
        return tok - 10, 1
    return 0, 0    # EOS/SEP/PAD


# ── Encoding ────────────────────────────────────────────────────────────────

def encode_addition(a: int, b: int, n_digits: int) -> dict:
    """
    Encode the addition problem  a + b  into token sequences.

    n_digits : number of digits in each operand (e.g. 3 → operands in 0..999)

    Returns a dict with:
        input_tokens  : list[int]   condition tokens (interleaved + SEP)
        output_tokens : list[int]   target tokens    (carry-encoded + EOS)
        n_pairs       : int         number of digit pairs in input = n_digits+1
        n_digits      : int         as passed in
        a, b, result  : int         the numbers
    """
    assert a >= 0 and b >= 0

    def to_digits(x, length):
        """Integer → list of digits LSB-first, padded to `length`."""
        out = []
        for _ in range(length):
            out.append(x % 10)
            x //= 10
        return out

    n_pairs = n_digits + 1               # extra pair for final carry step

    digits_a = to_digits(a, n_pairs)     # [a0, a1, ..., a_n]  LSB first, 0-padded
    digits_b = to_digits(b, n_pairs)

    # Interleaved input: [a0, b0, a1, b1, ..., a_n, b_n, SEP]
    input_tokens = []
    for i in range(n_pairs):
        input_tokens.append(digits_a[i])
        input_tokens.append(digits_b[i])
    input_tokens.append(SEP)

    # Output: carry-encoded tokens computed by the FSM
    result    = a + b
    carry     = 0
    output_tokens = []
    for i in range(n_pairs):
        ai  = digits_a[i]
        bi  = digits_b[i]
        s   = ai + bi + carry
        dig = s % 10
        carry = s // 10
        output_tokens.append(output_token(dig, carry))
    output_tokens.append(EOS)

    return {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "n_pairs":       n_pairs,
        "n_digits":      n_digits,
        "a": a, "b": b, "result": result,
    }


def decode_output(tokens: list[int]) -> int:
    """Decode a list of carry-encoded output tokens back to an integer."""
    value = 0
    for i, tok in enumerate(tokens):
        if tok == EOS or tok == PAD:
            break
        digit, _ = decode_output_token(tok)
        value += digit * (10 ** i)
    return value


# ── Quick sanity check ──────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(3, 7, 1), (99, 1, 2), (347, 658, 3)]
    for a, b, nd in tests:
        ex = encode_addition(a, b, nd)
        decoded = decode_output(ex["output_tokens"])
        status = "OK" if decoded == a + b else f"FAIL (got {decoded})"
        print(f"{a} + {b} = {a+b}")
        print(f"  input : {[TOKEN_TO_STR[t] for t in ex['input_tokens']]}")
        print(f"  output: {[TOKEN_TO_STR[t] for t in ex['output_tokens']]}")
        print(f"  decoded: {decoded}  {status}")
        print()
