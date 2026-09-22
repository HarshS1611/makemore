"""
Tests for bigram.py. Run:
    cd makemore_project && pytest tests/test_bigram.py -v

Run just one function's tests while building:
    pytest tests/test_bigram.py -v -k vocab
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from bigram import (
    build_vocab, count_bigrams, counts_to_probs, sample_counting,
    nll_loss_counting, build_training_set, train_gradient_descent,
)

WORDS = ["ab", "abc", "ba"]


def test_vocab_size_and_dot():
    stoi, itos = build_vocab(WORDS)
    # 3 letters (a, b, c) + the '.' token = 4
    assert len(stoi) == 4
    assert stoi['.'] == 0
    assert itos[0] == '.'


def test_vocab_roundtrip():
    stoi, itos = build_vocab(WORDS)
    for ch, ix in stoi.items():
        assert itos[ix] == ch


def test_count_bigrams_shape():
    stoi, itos = build_vocab(WORDS)
    N = count_bigrams(WORDS, stoi)
    assert N.shape == (len(stoi), len(stoi))


def test_count_bigrams_values():
    """Hand-counted: words are ab, abc, ba (bracketed with '.').
    .a  ab  b.        -> from "ab"
    .a  ab  bc  c.     -> from "abc"
    .b  ba  a.        -> from "ba"
    So bigram (a,b) appears twice: 'ab' in "ab" and 'ab' in "abc".
    """
    stoi, itos = build_vocab(WORDS)
    N = count_bigrams(WORDS, stoi)
    a, b = stoi['a'], stoi['b']
    assert N[a, b].item() == 2, "bigram 'ab' should occur exactly twice across the dataset"


def test_probs_rows_sum_to_one():
    stoi, itos = build_vocab(WORDS)
    N = count_bigrams(WORDS, stoi)
    P = counts_to_probs(N)
    row_sums = P.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(len(stoi)), atol=1e-5), (
        "every row of P must be a valid probability distribution "
        "(sum to 1) — check your normalization in counts_to_probs"
    )


def test_probs_no_zero_entries():
    """Smoothing check: if any entry of P is exactly 0, nll_loss_counting
    will compute log(0) = -inf on some real dataset, which breaks
    everything. This is why counting adds +1 before normalizing.
    """
    stoi, itos = build_vocab(WORDS)
    N = count_bigrams(WORDS, stoi)
    P = counts_to_probs(N)
    assert (P > 0).all(), (
        "P has zero entries — add smoothing (e.g. N.float() + 1) "
        "before normalizing in counts_to_probs"
    )


def test_sample_counting_produces_n_names():
    stoi, itos = build_vocab(WORDS)
    N = count_bigrams(WORDS, stoi)
    P = counts_to_probs(N)
    names = sample_counting(P, itos, n=5)
    assert len(names) == 5
    assert all(isinstance(n, str) for n in names)


def test_sample_counting_uses_only_known_letters():
    stoi, itos = build_vocab(WORDS)
    N = count_bigrams(WORDS, stoi)
    P = counts_to_probs(N)
    names = sample_counting(P, itos, n=20)
    known = set(stoi.keys()) - {'.'}
    for name in names:
        assert set(name).issubset(known), f"generated name {name!r} used a letter outside the vocab"


def test_nll_loss_is_positive_finite():
    stoi, itos = build_vocab(WORDS)
    N = count_bigrams(WORDS, stoi)
    P = counts_to_probs(N)
    loss = nll_loss_counting(WORDS, stoi, P)
    assert loss > 0
    assert loss == loss  # not NaN
    assert loss != float("inf")


def test_perfect_predictable_dataset_has_low_loss():
    """If EVERY word is identical, the counting model should predict
    it with near-certainty, so loss should be very close to 0 -- much
    lower than on the varied WORDS dataset above.
    """
    words = ["aaa"] * 50
    stoi, itos = build_vocab(words)
    N = count_bigrams(words, stoi)
    P = counts_to_probs(N)
    loss = nll_loss_counting(words, stoi, P)
    assert loss < 0.5, (
        f"expected near-zero loss on a fully predictable dataset, got {loss} "
        "— check counts_to_probs and nll_loss_counting"
    )


def test_build_training_set_shapes_match():
    stoi, itos = build_vocab(WORDS)
    xs, ys = build_training_set(WORDS, stoi)
    assert xs.shape == ys.shape
    assert xs.dtype in (torch.int64, torch.long)


def test_build_training_set_matches_bigram_count():
    """Total rows in xs/ys should equal the sum of all bigram counts."""
    stoi, itos = build_vocab(WORDS)
    N = count_bigrams(WORDS, stoi)
    xs, ys = build_training_set(WORDS, stoi)
    assert xs.shape[0] == N.sum().item()


def test_gradient_descent_loss_decreases():
    stoi, itos = build_vocab(WORDS)
    xs, ys = build_training_set(WORDS, stoi)
    W, losses = train_gradient_descent(xs, ys, steps=50, lr=10.0, vocab_size=len(stoi))
    assert losses[-1] < losses[0], "loss should decrease over training, even on this tiny dataset"


def test_gradient_descent_matches_counting_model():
    """THE integration test for Part 1: on the SAME data, the
    gradient-descent model should converge close to the counting
    model's loss. This is the entire point of the bigram exercise --
    two totally different training procedures, same answer.
    """
    stoi, itos = build_vocab(WORDS)
    N = count_bigrams(WORDS, stoi)
    P = counts_to_probs(N)
    counting_loss = nll_loss_counting(WORDS, stoi, P)

    xs, ys = build_training_set(WORDS, stoi)
    W, losses = train_gradient_descent(xs, ys, steps=200, lr=10.0, vocab_size=len(stoi))

    assert abs(losses[-1] - counting_loss) < 0.3, (
        f"gradient descent converged to {losses[-1]:.4f} but the counting "
        f"model's loss is {counting_loss:.4f} — these should be close. "
        "If they're not, check the smoothing term you use in the "
        "gradient-descent loss (a tiny regularizer changes the target "
        "slightly, but not by much) or your softmax normalization."
    )
