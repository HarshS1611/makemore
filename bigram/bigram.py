import os

import torch
import torch.nn.functional as F

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "names.txt")


def read_names(path=DATA_PATH):
    return open(path).read().splitlines()


def build_vocab(words):
    # index 0 is '.', which marks both the start and the end of a name
    stoi = {s: i for i, s in enumerate(['.'] + sorted(set(''.join(words))))}
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


# ---- counting model ----

def count_bigrams(words, stoi):
    N = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)
    for word in words:
        chars = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            N[stoi[ch1], stoi[ch2]] += 1
    return N


def counts_to_probs(N):
    P = (N + 1).float()  # +1 smoothing: no bigram gets probability 0
    P /= P.sum(dim=1, keepdim=True)
    return P


def sample_counting(P, itos, n=20, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(n):
        name = []
        ix = 0
        while True:
            ix = torch.multinomial(P[ix], num_samples=1, generator=g).item()
            if ix == 0:
                break
            name.append(itos[ix])
        names.append(''.join(name))
    return names


def nll_loss_counting(words, stoi, P):
    log_likelihood = 0.0
    n = 0
    for word in words:
        chars = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
            n += 1
    return -(log_likelihood / n).item()


# ---- same model, trained by gradient descent ----

def build_training_set(words, stoi):
    xs, ys = [], []
    for word in words:
        chars = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def train_gradient_descent(xs, ys, steps=100, lr=50.0, vocab_size=27, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)
    xenc = F.one_hot(xs, num_classes=vocab_size).float()
    losses = []
    for _ in range(steps):
        logits = xenc @ W
        counts = logits.exp()  # exp(logits) plays the role of the counts N
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(len(xs)), ys].log().mean()
        loss = loss + 0.01 * (W**2).mean()  # regularization, the counterpart of +1 smoothing

        W.grad = None
        loss.backward()
        with torch.no_grad():
            W -= lr * W.grad
        losses.append(loss.item())
    return W, losses


if __name__ == "__main__":
    words = read_names()
    stoi, itos = build_vocab(words)

    N = count_bigrams(words, stoi)
    P = counts_to_probs(N)
    print(f"counting model loss:         {nll_loss_counting(words, stoi, P):.4f}")
    print("samples:", sample_counting(P, itos, n=10))

    xs, ys = build_training_set(words, stoi)
    W, losses = train_gradient_descent(xs, ys, steps=100)
    print(f"gradient descent final loss: {losses[-1]:.4f}")
