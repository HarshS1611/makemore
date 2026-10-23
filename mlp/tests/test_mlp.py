import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from mlp import (
    build_vocab, build_dataset, split_dataset, init_params, forward,
    train, split_loss, sample, BLOCK_SIZE,
)

WORDS = ["emma", "olivia", "ava", "isabella", "sophia", "mia", "amelia",
         "harper", "evelyn", "abigail"] * 3


def _split():
    stoi, itos = build_vocab(WORDS)
    return stoi, itos, split_dataset(WORDS, stoi, block_size=BLOCK_SIZE)


def test_vocab_size():
    stoi, _ = build_vocab(WORDS)
    assert stoi['.'] == 0
    assert len(stoi) == len(set(''.join(WORDS))) + 1


def test_build_dataset_shapes():
    stoi, _ = build_vocab(WORDS)
    X, Y = build_dataset(WORDS, stoi, block_size=3)
    assert X.shape[1] == 3 and X.shape[0] == Y.shape[0]
    assert X.dtype == torch.long


def test_build_dataset_known_example():
    # "ab" with block_size=3:  ... -> a,  ..a -> b,  .ab -> .
    stoi, _ = build_vocab(["ab"])
    X, Y = build_dataset(["ab"], stoi, block_size=3)
    dot, a, b = stoi['.'], stoi['a'], stoi['b']
    pairs = {tuple(x.tolist()): y for x, y in zip(X, Y.tolist())}
    assert pairs == {(dot, dot, dot): a, (dot, dot, a): b, (dot, a, b): dot}


def test_split_dataset_proportions():
    _, _, (Xtr, _, Xdev, _, Xte, _) = _split()
    total = Xtr.shape[0] + Xdev.shape[0] + Xte.shape[0]
    assert 0.7 < Xtr.shape[0] / total < 0.9


def test_split_dataset_does_not_mutate_input():
    before = list(WORDS)
    stoi, _ = build_vocab(WORDS)
    split_dataset(WORDS, stoi)
    assert WORDS == before


def test_init_params_shapes():
    p = init_params(vocab_size=27, block_size=3, n_embd=10, n_hidden=200)
    assert p['C'].shape == (27, 10)
    assert p['W1'].shape == (30, 200)
    assert p['b1'].shape == (200,)
    assert p['W2'].shape == (200, 27)
    assert p['b2'].shape == (27,)


def test_init_params_are_trainable_leaves():
    # a tensor built as `randn(requires_grad=True) * s` is not a leaf and never gets .grad
    for name, p in init_params().items():
        assert p.requires_grad and p.is_leaf, name


def test_forward_logits_shape():
    p = init_params(vocab_size=27, block_size=3, n_embd=10, n_hidden=200)
    logits, loss = forward(p, torch.zeros((5, 3), dtype=torch.long))
    assert logits.shape == (5, 27) and loss is None


def test_forward_loss_is_scalar_and_finite():
    p = init_params(vocab_size=27, block_size=3, n_embd=10, n_hidden=200)
    X = torch.zeros((5, 3), dtype=torch.long)
    _, loss = forward(p, X, torch.zeros((5,), dtype=torch.long))
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_forward_loss_near_uniform_at_init():
    # untrained over ~27 classes: about -log(1/27) = 3.3; hundreds means the init scale is too large
    stoi, _ = build_vocab(WORDS)
    X, Y = build_dataset(WORDS, stoi, block_size=BLOCK_SIZE)
    _, loss = forward(init_params(vocab_size=len(stoi), block_size=BLOCK_SIZE), X, Y)
    assert 2.0 < loss.item() < 6.0


def test_backward_reaches_embedding_table():
    stoi, _ = build_vocab(WORDS)
    X, Y = build_dataset(WORDS, stoi, block_size=BLOCK_SIZE)
    p = init_params(vocab_size=len(stoi), block_size=BLOCK_SIZE)
    forward(p, X, Y)[1].backward()
    assert p['C'].grad is not None and p['C'].grad.abs().sum().item() > 0


def test_train_loss_decreases():
    stoi, _, (Xtr, Ytr, *_) = _split()
    p = init_params(vocab_size=len(stoi), block_size=BLOCK_SIZE, n_hidden=50)
    losses = train(p, Xtr, Ytr, steps=300, batch_size=16, lr=0.1)
    assert sum(losses[-20:]) / 20 < sum(losses[:20]) / 20


def test_split_loss_returns_positive_float():
    stoi, _, (_, _, Xdev, Ydev, *_) = _split()
    value = split_loss(init_params(vocab_size=len(stoi), block_size=BLOCK_SIZE), Xdev, Ydev)
    assert isinstance(value, float) and value > 0


def test_sample_produces_n_names_of_known_characters():
    stoi, itos, _ = _split()
    p = init_params(vocab_size=len(stoi), block_size=BLOCK_SIZE, n_hidden=50)
    names = sample(p, itos, n=5)
    assert len(names) == 5
    assert all(set(name) <= set(stoi) - {'.'} for name in names)
