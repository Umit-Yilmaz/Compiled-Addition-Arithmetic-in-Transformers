"""
model.py — Hand-compiled transformer for exact integer addition.

════════════════════════════════════════════════════════════════
ARCHITECTURE OVERVIEW
════════════════════════════════════════════════════════════════

  1 transformer layer  (3 attention heads + MLP)
  Hand-compiled weights — no training, no gradient descent.

  Why 1 layer is sufficient (key insight):
    Carry is encoded in the OUTPUT TOKEN ID (tokens 10-19 = carry=1).
    Head 2 reads carry from the PREVIOUS TOKEN'S EMBEDDING — accessible
    at attention time, no MLP needed. This breaks the "depth = n" barrier.

════════════════════════════════════════════════════════════════
EMBEDDING LAYOUT  (D_MODEL = 46)
════════════════════════════════════════════════════════════════

  [0  : 10]  step one-hot       which addition step this token belongs to
  [10 : 20]  a-slot             a-digit one-hot (for input), answer one-hot (for output)
  [20 : 30]  b-slot             b-digit one-hot
  [30]        is_a              1 if a-digit input token
  [31]        is_b              1 if b-digit input token
  [32]        is_out            1 if output digit token
  [33]        is_sep            1 if SEP
  [34]        carry             carry_out encoded in token (0 or 1)
  [35]        bias              always 1

════════════════════════════════════════════════════════════════
ATTENTION HEADS  (3 heads, D_H = 12)
════════════════════════════════════════════════════════════════

  HEAD 0 — a-digit retrieval
    Attends to the a-digit token at the SAME STEP as the current output.
    Q = [step_one_hot(i) * S, GATE at dim 10]  is_out query
    K = [step_one_hot(j) * S, GATE at dim 10]  is_a  key
    → score = S² * delta(i==j)  +  GATE² * is_out_i * is_a_j  (exact)

  HEAD 1 — b-digit retrieval (same, but IS_B gate)

  HEAD 2 — carry retrieval (relative position −1)
    Attends to the PREVIOUS output token to read carry from its embedding.
    Q = [step_one_hot(i)   * S, GATE at dim 11]  unshifted
    K = [step_one_hot(j+1) * S, GATE at dim 11]  SHIFTED by +1
    → score = S² * delta(i == j+1)  +  GATE²   for output-output pairs
    → attends to j = i−1  ✓
    V reads CARRY from the token's embedding (set by embed matrix for tokens 10-19)

════════════════════════════════════════════════════════════════
MLP  (200 neurons — one per (a ∈ 0..9, b ∈ 0..9, c ∈ {0,1}) triple)
════════════════════════════════════════════════════════════════

  After attention the output token's residual has:
    a-slot [10:20] : aᵢ one-hot  (from head 0)
    b-slot [20:30] : bᵢ one-hot  (from head 1)
    carry  [34]    : carry_in    (from head 2 reading prev token's embedding)

  Layer 1 (hidden=200, ReLU):  each neuron detects one (a, b, carry) triple
  Layer 2 (linear):            writes answer digit and carry_out

  carry=0 neuron for (a, b):  W[a-slot+a]=1, W[b-slot+b]=1, W[CARRY]=-2, W[IS_OUT]=2  bias=-3.5
  carry=1 neuron for (a, b):  W[a-slot+a]=1, W[b-slot+b]=1, W[CARRY]=+2, W[IS_OUT]=2  bias=-5.5

  Layer 2 output = delta carry + delta digit (via residual cancellation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vocab import VOCAB_SIZE, SEP, EOS, PAD, DIGITS_IN, output_token, decode_output_token

# ── Constants ────────────────────────────────────────────────────────────────

N_MAX      = 10      # max digit pairs  (supports up to 10-digit operands)
D_MODEL    = 36      # embedding dimension

# Slice / index layout (D_MODEL = 10+10+10+6 = 36)
STEP_SL    = slice(0, 10)       # [0:10]  step one-hot
A_SL       = slice(10, 20)      # [10:20] a-digit one-hot / answer digit
B_SL       = slice(20, 30)      # [20:30] b-digit one-hot
IS_A       = 30
IS_B       = 31
IS_OUT     = 32
IS_SEP     = 33
CARRY      = 34
BIAS       = 35

N_HEADS    = 3
D_H        = 12      # dims per head  (needs ≥ N_MAX + 2 = 12  ✓)
HIDDEN     = 200     # MLP hidden units

# Attention score parameters.
# Score = STEP_SCALE² (step match) + GATE² (type match)
#   correct token: STEP_SCALE² + GATE² = 20000  (wins decisively)
#   wrong step:    0            + GATE² = 10000  ← negligible after softmax
#   wrong type:    STEP_SCALE²  + 0     = 10000  ← negligible after softmax
#   both wrong:    0            + 0     = 0
# softmax([20000, 10000, ..., 10000]):
#   exp(20000-20000) / (1 + N * exp(-10000)) ≈ 1.0  ✓
STEP_SCALE = 100.0
GATE       = 100.0


# ── Compiled transformer ─────────────────────────────────────────────────────

class CompiledAdditionTransformer(nn.Module):
    """
    1-layer transformer with hand-compiled weights implementing n-digit addition.

    All parameters have requires_grad=False.
    Compared against an identically-shaped trained model in src/trained/model.py.
    """

    def __init__(self):
        super().__init__()

        # ── Token embedding ────────────────────────────────────────────────
        # What we know from token ID alone (role added by positional encoding):
        #   tokens 0-9  : digit v, carry=0  → a-slot[v]=1
        #   tokens 10-19: digit v, carry=1  → a-slot[v]=1, CARRY=1
        #   SEP         : is_sep=1
        #   EOS, PAD    : all zeros
        E = torch.zeros(VOCAB_SIZE, D_MODEL)
        for v in range(10):
            E[v,      10 + v] = 1.0    # carry=0 output tokens
            E[v,      BIAS]   = 1.0
            E[10 + v, 10 + v] = 1.0    # carry=1 output tokens (same digit slot)
            E[10 + v, CARRY]  = 1.0    # carry bit = 1  ← key design choice
            E[10 + v, BIAS]   = 1.0
        E[SEP, IS_SEP] = 1.0
        E[SEP, BIAS]   = 1.0
        E[EOS, BIAS]   = 1.0
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.embedding.weight = nn.Parameter(E, requires_grad=False)

        # ── Attention weights (N_HEADS, D_MODEL, D_H) ────────────────────
        self.W_Q = nn.Parameter(torch.zeros(N_HEADS, D_MODEL, D_H), requires_grad=False)
        self.W_K = nn.Parameter(torch.zeros(N_HEADS, D_MODEL, D_H), requires_grad=False)
        self.W_V = nn.Parameter(torch.zeros(N_HEADS, D_MODEL, D_H), requires_grad=False)
        self.W_O = nn.Parameter(torch.zeros(N_HEADS, D_H, D_MODEL), requires_grad=False)
        self._compile_attention()

        # ── MLP weights ───────────────────────────────────────────────────
        self.W1 = nn.Parameter(torch.zeros(HIDDEN, D_MODEL), requires_grad=False)
        self.b1 = nn.Parameter(torch.zeros(HIDDEN),          requires_grad=False)
        self.W2 = nn.Parameter(torch.zeros(D_MODEL, HIDDEN), requires_grad=False)
        self.b2 = nn.Parameter(torch.zeros(D_MODEL),         requires_grad=False)
        self._compile_mlp()

        # ── Output projection ─────────────────────────────────────────────
        # After MLP, a-slot [10:20] contains the answer-digit one-hot,
        # and CARRY [34] contains the new carry bit.
        # We project these to logits over VOCAB_SIZE.
        out = torch.zeros(D_MODEL, VOCAB_SIZE)
        for v in range(10):
            out[10 + v, v]      = 20.0   # digit v, carry=0  → token v
            out[10 + v, 10 + v] = 20.0   # digit v, carry=1  → token 10+v
        # The CARRY bit selects between token v (carry=0) and 10+v (carry=1).
        # We add CARRY contribution: if carry=1 and digit=v → 10+v wins.
        for v in range(10):
            out[CARRY, 10 + v] =  20.0   # CARRY=1 boosts token 10+v
            out[CARRY, v]      = -20.0   # CARRY=1 suppresses token v
        out[IS_SEP, SEP] = 20.0
        out[BIAS,   EOS] = 5.0
        self.out_proj = nn.Parameter(out, requires_grad=False)

    # ── Compile attention weights ────────────────────────────────────────────

    def _compile_attention(self):
        """Set Q, K, V, O for all 3 heads using exact one-hot step matching."""

        # ── Heads 0 and 1: digit retrieval ────────────────────────────────
        for head, (IS_X, V_SL) in enumerate([(IS_A, A_SL), (IS_B, B_SL)]):
            WQ = self.W_Q[head]
            WK = self.W_K[head]
            WV = self.W_V[head]
            WO = self.W_O[head]

            # Step matching: W_Q[step, step] = W_K[step, step] = STEP_SCALE
            # → Q·K from step one-hots = STEP_SCALE² if same step, else 0
            for step in range(N_MAX):
                WQ[step, step] = STEP_SCALE
                WK[step, step] = STEP_SCALE

            # Type gate at head dim N_MAX (= dim 10):
            # Output tokens ASK  (W_Q[IS_OUT, 10] = GATE)
            # Target-type tokens ANSWER  (W_K[IS_X, 10] = GATE)
            WQ[IS_OUT, N_MAX]  = GATE
            WK[IS_X,   N_MAX]  = GATE

            # Value: read digit one-hot from the target slot, write to A_SL
            for v in range(10):
                WV[V_SL.start + v, v] = 1.0     # slot one-hot → head dim v
                WO[v, A_SL.start + v] = 1.0     # head dim v → a-slot[v]
            # Head 1 writes to b-slot instead:
            if head == 1:
                for v in range(10):
                    WO[v, A_SL.start + v] = 0.0          # undo a-slot write
                    WO[v, B_SL.start + v] = 1.0          # write to b-slot

        # ── Head 2: carry (relative position −1) ──────────────────────────
        #
        # SHIFTED KEY trick:
        #   Q at step i: STEP_SCALE at dim i        (unshifted, dims 0..9)
        #   K at step j: STEP_SCALE at dim j+1      (shifted,   dims 1..10)
        #   → Q·K = STEP_SCALE² iff i == j+1  (i.e., j == i−1)
        #
        # Type gate at dim 11 (out of range 0..10):
        #   Q[IS_OUT, 11] = GATE, K[IS_OUT, 11] = GATE
        #   → only output-output pairs get +GATE²
        #
        # Step i=0: no j=−1 exists → score=GATE² for self, 0 for inputs.
        #   Attends to self with carry=0 (initial embedding has CARRY=0). ✓
        #   (Self-carry is 0 for the first output token since c₀ hasn't been
        #   generated yet — the dummy token defaults to carry=0 in embedding.)

        WQ = self.W_Q[2]
        WK = self.W_K[2]
        WV = self.W_V[2]
        WO = self.W_O[2]

        for step in range(N_MAX):
            WQ[step,      step]     = STEP_SCALE    # unshifted Q
            WK[step,      step + 1] = STEP_SCALE    # shifted   K  (dim j+1)

        WQ[IS_OUT, N_MAX + 1] = GATE    # type gate at dim 11
        WK[IS_OUT, N_MAX + 1] = GATE

        WV[CARRY, 0]   = 1.0    # read carry from token embedding → head dim 0
        WO[0,  CARRY]  = 1.0    # head dim 0 → carry slot [34]

    # ── Compile MLP weights ──────────────────────────────────────────────────

    def _compile_mlp(self):
        """
        Build lookup-table MLP: (a, b, carry_in) → (digit_out, carry_out).

        200 neurons, one per (a ∈ 0..9, b ∈ 0..9, c ∈ {0,1}) triple.

        Each neuron fires with activation 0.5 for exactly its triple (the
        indicator-pair construction: ReLU(total - s + 0.5) where s is the
        threshold).  Layer 2 multiplies by 2 to get contribution = 1.0.

        Layer 2 also cancels the OLD a-digit (from attention) via residual,
        since after MLP the a-slot must hold the ANSWER digit, not aᵢ.
        """
        W1, b1, W2 = self.W1.data, self.b1.data, self.W2.data

        for a in range(10):
            for b in range(10):
                for c in range(2):
                    n = 2 * (10 * a + b) + c    # neuron index 0..199

                    # ── Layer 1: detect (a, b, c) ──────────────────────────
                    W1[n, A_SL.start + a] = 1.0   # read a from a-slot
                    W1[n, B_SL.start + b] = 1.0   # read b from b-slot
                    W1[n, IS_OUT]         = 2.0   # is_out gate (prevents firing on input tokens)

                    if c == 0:
                        W1[n, CARRY] = -2.0    # penalize carry=1
                        b1[n]        = -3.5    # threshold
                        # fires: 1(a) + 1(b) + 0(carry) + 2(is_out) - 3.5 = 0.5 > 0  ✓
                        # silent: carry=1 → 1+1-2+2-3.5 = -0.5 < 0                   ✓
                        # silent: is_out=0 → 1+1+0+0-3.5 = -1.5 < 0                 ✓
                    else:
                        W1[n, CARRY] = 2.0     # require carry=1
                        b1[n]        = -5.5    # threshold
                        # fires: 1+1+2+2-5.5 = 0.5 > 0                               ✓

                    # ── Layer 2: write result via residual cancellation ────
                    #   Neuron fires with value 0.5.  ×2 gives contribution 1.0.
                    #   After residual (x += MLP(x)):
                    #     old a-slot[a] was 1 (from attention head 0).
                    #     Write +digit − old_a to make a-slot = answer one-hot.
                    total     = a + b + c
                    digit     = total % 10
                    carry_out = total // 10

                    W2[A_SL.start + digit, n] += 2.0    # +new digit
                    W2[A_SL.start + a,     n] -= 2.0    # −old a (residual cancel)

                    W2[CARRY, n] += 2.0 * carry_out     # +new carry
                    W2[CARRY, n] -= 2.0 * c             # −old carry (residual cancel)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def _positional_encoding(self, tokens: torch.Tensor, n_pairs: int) -> torch.Tensor:
        """
        Build the position-dependent additive annotation.

        For a-digit at pos 2i:   step=i, is_a=1
        For b-digit at pos 2i+1: step=i, is_b=1, move value from a-slot to b-slot
        For output at pos L+i:   step=i, is_out=1, clear a-slot from token embedding
        """
        seq_len   = tokens.shape[0]
        input_len = 2 * n_pairs + 1
        P = torch.zeros(seq_len, D_MODEL)

        for pos in range(seq_len):
            tok = tokens[pos].item()

            if pos < 2 * n_pairs:
                step = pos // 2
                P[pos, step] = 1.0          # step one-hot

                if pos % 2 == 0:            # a-digit
                    P[pos, IS_A] = 1.0

                else:                        # b-digit
                    P[pos, IS_B] = 1.0
                    if tok < 10:            # digit token: move a-slot → b-slot
                        P[pos, A_SL.start + tok] = -1.0   # cancel a-slot
                        P[pos, B_SL.start + tok] = +1.0   # fill b-slot

            elif pos == 2 * n_pairs:
                pass                        # SEP: embedding already has is_sep

            else:                           # output region
                step = pos - input_len
                P[pos, step]   = 1.0        # step one-hot
                P[pos, IS_OUT] = 1.0
                # Clear a-slot: the digit from the token embedding is noise here.
                # Head 0 will write the correct aᵢ value into the a-slot.
                # (The carry bit in CARRY is kept — it came from the token embedding
                #  and is correct for carry reading by head 2.)
                if tok < 20:               # any digit/carry output token
                    v = tok % 10           # the digit value in the token
                    P[pos, A_SL.start + v] = -1.0   # cancel it

        return P

    def forward(self, tokens: torch.Tensor, n_pairs: int) -> torch.Tensor:
        """
        tokens  : (seq_len,) int64
        n_pairs : number of digit pairs in this problem (= n_digits + 1)
        Returns : (seq_len, VOCAB_SIZE) logits
        """
        seq_len = tokens.shape[0]

        # 1. Embed + positional encoding
        x = self.embedding(tokens) + self._positional_encoding(tokens, n_pairs)

        # 2. Causal attention mask  (upper triangle = −∞)
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf')), diagonal=1
        )

        # 3. Multi-head self-attention (with residual)
        attn_out = torch.zeros_like(x)
        for h in range(N_HEADS):
            Q = x @ self.W_Q[h]                    # (seq_len, D_H)
            K = x @ self.W_K[h]
            V = x @ self.W_V[h]
            scores  = Q @ K.T + attn_mask          # (seq_len, seq_len)
            weights = F.softmax(scores, dim=-1)
            attn_out += (weights @ V) @ self.W_O[h]

        x = x + attn_out                           # residual

        # 4. MLP (with residual)
        h1 = F.relu(x @ self.W1.T + self.b1)      # (seq_len, HIDDEN)
        x  = x + h1 @ self.W2.T + self.b2         # residual

        # 5. Output logits
        return x @ self.out_proj                   # (seq_len, VOCAB_SIZE)

    # ── Generation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, a: int, b: int, n_digits: int) -> int:
        """
        Compute  a + b  using the compiled model.
        Returns the decoded integer result.
        """
        from .vocab import encode_addition, decode_output

        ex           = encode_addition(a, b, n_digits)
        input_tokens = ex["input_tokens"]
        n_pairs      = ex["n_pairs"]
        n_out        = n_pairs                  # one output token per pair

        tokens = input_tokens[:]

        for _ in range(n_out):
            # Append a dummy token (token id 0 = digit 0, carry=0).
            # The a-slot from its embedding is cancelled by positional encoding,
            # so it doesn't contaminate the computation.
            tokens.append(0)
            t      = torch.tensor(tokens, dtype=torch.long)
            pred   = self.forward(t, n_pairs)[-1].argmax().item()
            tokens[-1] = pred                   # replace dummy with prediction

        tokens.append(EOS)
        return decode_output(tokens[len(input_tokens):])


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = CompiledAdditionTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}  (all frozen)")

    tests = [
        (0,       0,    1),
        (3,       7,    1),    # 10
        (9,       9,    1),    # 18
        (99,      1,    2),    # 100
        (347,     658,  3),    # 1005
        (9999,    1,    4),    # 10000
        (12345,   67890, 5),   # 80235
        (9999999, 1,    7),    # 10000000
    ]

    all_ok = True
    for a, b, nd in tests:
        result   = model.generate(a, b, nd)
        expected = a + b
        ok       = result == expected
        status   = "OK" if ok else f"FAIL (got {result})"
        print(f"  {a:>10} + {b:<10} = {result:<12}  {status}")
        all_ok &= ok

    print()
    print("All tests passed." if all_ok else "Some tests FAILED.")
