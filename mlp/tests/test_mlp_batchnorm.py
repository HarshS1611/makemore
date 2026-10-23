import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from mlp import build_vocab, build_dataset, split_dataset, BLOCK_SIZE
from mlp_batchnorm import init_params_bn, forward_bn, train_bn, split_loss_bn

WORDS = ["emma", "olivia", "ava", "isabella", "sophia", "mia", "amelia",
         "harper", "evelyn", "abigail"] * 3
LEARNABLE = ('C', 'W1', 'W2', 'b2', 'bngain', 'bnbias')
RUNNING = ('bnmean_running', 'bnvar_running')


def _small(n_hidden=50):
    stoi, _ = build_vocab(WORDS)
    Xtr, Ytr, *_ = split_dataset(WORDS, stoi, block_size=BLOCK_SIZE)
    return init_params_bn(vocab_size=len(stoi), block_size=BLOCK_SIZE, n_hidden=n_hidden), Xtr, Ytr


def test_init_params_bn_shapes():
    p = init_params_bn(vocab_size=27, block_size=3, n_embd=10, n_hidden=200)
    assert p['C'].shape == (27, 10)
    assert p['W1'].shape == (30, 200)
    assert p['W2'].shape == (200, 27)
    assert p['b2'].shape == (27,)
    for name in ('bngain', 'bnbias', 'bnmean_running', 'bnvar_running'):
        assert p[name].shape == (1, 200)


def test_init_params_bn_has_no_b1():
    # batchnorm subtracts the batch mean right after the linear layer, so a bias would cancel
    assert 'b1' not in init_params_bn()


def test_learnable_params_are_trainable_leaves():
    # a tensor built as `x * s` after `requires_grad=True` is not a leaf and never gets .grad
    p = init_params_bn()
    for name in LEARNABLE:
        assert p[name].requires_grad and p[name].is_leaf, name


def test_running_stats_do_not_require_grad():
    # they are updated by a running average, never by .backward()
    p = init_params_bn()
    for name in RUNNING:
        assert not p[name].requires_grad, name


def test_running_stats_init_values():
    p = init_params_bn(n_hidden=50)
    assert torch.allclose(p['bnmean_running'], torch.zeros(1, 50))
    assert torch.allclose(p['bnvar_running'], torch.ones(1, 50))


def test_forward_bn_logits_shape():
    p = init_params_bn(vocab_size=27, block_size=3, n_embd=10, n_hidden=200)
    logits, loss = forward_bn(p, torch.zeros((5, 3), dtype=torch.long))
    assert logits.shape == (5, 27) and loss is None


def test_forward_bn_is_invariant_to_w1_scale():
    # normalization removes the scale of the pre-activations, so rescaling W1 changes nothing
    p, Xtr, Ytr = _small()
    logits_a, _ = forward_bn(p, Xtr[:64], Ytr[:64], training=True)
    with torch.no_grad():
        p['W1'] *= 20
    logits_b, _ = forward_bn(p, Xtr[:64], Ytr[:64], training=True)
    assert torch.allclose(logits_a, logits_b, atol=1e-3)


def test_forward_bn_eval_uses_running_stats_not_batch_stats():
    # at eval time the output for one example must not depend on what batch it sits in
    torch.manual_seed(0)
    p = init_params_bn(vocab_size=27, block_size=3, n_embd=10, n_hidden=50)
    with torch.no_grad():
        p['bnmean_running'] += 5.0
        p['bnvar_running'] *= 3.0
    single, _ = forward_bn(p, torch.zeros((1, 3), dtype=torch.long), training=False)
    repeated, _ = forward_bn(p, torch.zeros((8, 3), dtype=torch.long), training=False)
    assert torch.allclose(single[0], repeated[0], atol=1e-4)


def test_running_stats_update_during_training():
    p, Xtr, Ytr = _small()
    before = p['bnmean_running'].clone()
    train_bn(p, Xtr, Ytr, steps=5, batch_size=16)
    assert not torch.allclose(p['bnmean_running'], before)


def test_train_bn_loss_decreases():
    p, Xtr, Ytr = _small()
    losses = train_bn(p, Xtr, Ytr, steps=300, batch_size=16)
    assert sum(losses[-20:]) / 20 < sum(losses[:20]) / 20


def test_train_bn_only_updates_learnable_params_with_gradients():
    # the running stats have no .grad; updating them with `p -= lr * p.grad` would raise
    p, Xtr, Ytr = _small()
    train_bn(p, Xtr, Ytr, steps=5, batch_size=16)
    assert all(p[name].grad is not None for name in LEARNABLE)
    assert all(p[name].grad is None for name in RUNNING)


def test_split_loss_bn_returns_positive_float():
    p, _, _ = _small()
    stoi, _ = build_vocab(WORDS)
    _, _, Xdev, Ydev, *_ = split_dataset(WORDS, stoi, block_size=BLOCK_SIZE)
    value = split_loss_bn(p, Xdev, Ydev)
    assert isinstance(value, float) and value > 0
