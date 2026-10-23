import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from bigram import (
    build_vocab, count_bigrams, counts_to_probs, sample_counting,
    nll_loss_counting, build_training_set, train_gradient_descent,
)

# bracketed bigrams: .a ab b. | .a ab bc c. | .b ba a.
WORDS = ["ab", "abc", "ba"]


def _probs(words=WORDS):
    stoi, itos = build_vocab(words)
    return stoi, itos, counts_to_probs(count_bigrams(words, stoi))


def test_vocab_size_and_dot():
    stoi, itos = build_vocab(WORDS)
    assert len(stoi) == 4
    assert stoi['.'] == 0 and itos[0] == '.'


def test_vocab_roundtrip():
    stoi, itos = build_vocab(WORDS)
    assert all(itos[ix] == ch for ch, ix in stoi.items())


def test_count_bigrams_shape():
    stoi, _ = build_vocab(WORDS)
    assert count_bigrams(WORDS, stoi).shape == (len(stoi), len(stoi))


def test_count_bigrams_values():
    stoi, _ = build_vocab(WORDS)
    N = count_bigrams(WORDS, stoi)
    assert N[stoi['a'], stoi['b']].item() == 2


def test_probs_rows_sum_to_one():
    stoi, _, P = _probs()
    assert torch.allclose(P.sum(dim=1), torch.ones(len(stoi)), atol=1e-5)


def test_probs_no_zero_entries():
    # a zero probability makes the loss log(0) = -inf
    _, _, P = _probs()
    assert (P > 0).all()


def test_sample_counting_produces_n_names():
    _, itos, P = _probs()
    names = sample_counting(P, itos, n=5)
    assert len(names) == 5 and all(isinstance(n, str) for n in names)


def test_sample_counting_uses_only_known_letters():
    stoi, itos, P = _probs()
    known = set(stoi) - {'.'}
    assert all(set(name) <= known for name in sample_counting(P, itos, n=20))


def test_nll_loss_is_positive_finite():
    stoi, _, P = _probs()
    loss = nll_loss_counting(WORDS, stoi, P)
    assert 0 < loss < float("inf")


def test_perfect_predictable_dataset_has_low_loss():
    words = ["aaa"] * 50
    stoi, _, P = _probs(words)
    assert nll_loss_counting(words, stoi, P) < 0.5


def test_build_training_set_shapes_match():
    stoi, _ = build_vocab(WORDS)
    xs, ys = build_training_set(WORDS, stoi)
    assert xs.shape == ys.shape and xs.dtype == torch.long


def test_build_training_set_matches_bigram_count():
    stoi, _ = build_vocab(WORDS)
    xs, _ = build_training_set(WORDS, stoi)
    assert xs.shape[0] == count_bigrams(WORDS, stoi).sum().item()


def test_gradient_descent_loss_decreases():
    stoi, _ = build_vocab(WORDS)
    xs, ys = build_training_set(WORDS, stoi)
    _, losses = train_gradient_descent(xs, ys, steps=50, lr=10.0, vocab_size=len(stoi))
    assert losses[-1] < losses[0]


def test_gradient_descent_matches_counting_model():
    # two different training methods should reach about the same loss
    stoi, _, P = _probs()
    counting_loss = nll_loss_counting(WORDS, stoi, P)
    xs, ys = build_training_set(WORDS, stoi)
    _, losses = train_gradient_descent(xs, ys, steps=200, lr=10.0, vocab_size=len(stoi))
    assert abs(losses[-1] - counting_loss) < 0.3
