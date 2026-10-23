import os
import random

import torch
import torch.nn.functional as F

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "names.txt")
BLOCK_SIZE = 10  # how many previous characters the model sees


def read_names(path=DATA_PATH):
    return open(path).read().splitlines()


def build_vocab(words):
    # index 0 is '.', which marks both the start and the end of a name
    stoi = {s: i for i, s in enumerate(['.'] + sorted(set(''.join(words))))}
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def build_dataset(words, stoi, block_size=BLOCK_SIZE):
    X, Y = [], []
    for word in words:
        context = [0] * block_size
        for ch in word + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


def split_dataset(words, stoi, block_size=BLOCK_SIZE, seed=42):
    words = words.copy()  # shuffle a copy, not the caller's list
    random.seed(seed)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    Xtr, Ytr = build_dataset(words[:n1], stoi, block_size)
    Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size)
    Xte, Yte = build_dataset(words[n2:], stoi, block_size)
    return Xtr, Ytr, Xdev, Ydev, Xte, Yte


def init_params(vocab_size=27, block_size=BLOCK_SIZE, n_embd=10, n_hidden=200, seed=2147483647):
    gen = lambda: torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, n_embd), generator=gen())
    W1 = torch.randn((block_size * n_embd, n_hidden), generator=gen()) * 0.1
    b1 = torch.randn((n_hidden,), generator=gen())
    W2 = torch.randn((n_hidden, vocab_size), generator=gen()) * 0.1
    b2 = torch.randn((vocab_size,), generator=gen())
    params = {'C': C, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    # enable gradients only after scaling: `randn(requires_grad=True) * s` is not a leaf
    for p in params.values():
        p.requires_grad_(True)
    return params


def forward(params, X, Y=None):
    emb = params['C'][X]                                    # (N, block_size, n_embd)
    emb_flat = emb.view(emb.shape[0], -1)                   # (N, block_size * n_embd)
    h = torch.tanh(emb_flat @ params['W1'] + params['b1'])  # (N, n_hidden)
    logits = h @ params['W2'] + params['b2']                # (N, vocab_size)
    loss = F.cross_entropy(logits, Y) if Y is not None else None
    return logits, loss


def train(params, Xtr, Ytr, steps=20000, batch_size=400, seed=2147483647, log_every=0):
    losses = []
    for step in range(steps):
        g = torch.Generator().manual_seed(seed + step)
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        _, loss = forward(params, Xtr[ix], Ytr[ix])

        for p in params.values():
            p.grad = None
        loss.backward()

        lr = 0.1 if step < 10000 else 0.01  # lower the learning rate for the second half
        for p in params.values():
            p.data += -lr * p.grad

        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"step {step:6d}/{steps} | loss {loss.item():.4f}")
    return losses


@torch.no_grad()
def split_loss(params, X, Y):
    _, loss = forward(params, X, Y)
    return loss.item()


def sample(params, itos, n=20, block_size=BLOCK_SIZE, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(n):
        context = [0] * block_size
        name = []
        while True:
            logits, _ = forward(params, torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, 1, generator=g).item()
            if ix == 0:
                break
            name.append(itos[ix])
            context = context[1:] + [ix]
        names.append(''.join(name))
    return names


if __name__ == "__main__":
    words = read_names()
    stoi, itos = build_vocab(words)
    Xtr, Ytr, Xdev, Ydev, Xte, Yte = split_dataset(words, stoi)

    params = init_params()
    print(f"dataset : {len(words):,} names, vocab {len(stoi)}")
    print(f"examples: train {Xtr.shape[0]:,} | dev {Xdev.shape[0]:,} | test {Xte.shape[0]:,}")
    print(f"params  : {sum(p.nelement() for p in params.values()):,}")
    print("-" * 40)

    train(params, Xtr, Ytr, steps=20000, log_every=2000)

    print("-" * 40)
    print(f"train loss: {split_loss(params, Xtr, Ytr):.4f}")
    print(f"dev loss  : {split_loss(params, Xdev, Ydev):.4f}")
    print(f"test loss : {split_loss(params, Xte, Yte):.4f}")
    print("samples:", sample(params, itos, n=10))
