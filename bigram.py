"""
bigram.py — a character-level bigram language model, trained two ways.

A bigram model predicts the next character from only the current one.
It's the simplest possible language model, and small enough to see
every number: 27x27 (26 letters + a boundary token), not millions of
parameters.

WHAT YOU'RE BUILDING
    1. build_vocab   — map each character to an integer and back
    2. count_bigrams  — build P(next char | current char) by counting
    3. sample_counting — generate names from the counting model
    4. nll_loss_counting — measure how good the counting model is
    5. train_gradient_descent — the SAME model, trained as a 1-layer
       net with a softmax, using torch and gradient descent
    6. Compare the two. They should converge to nearly the same loss —
       that's the actual point of Part 1.

DATA
    names.txt, one name per line, lowercase. Each name is treated as a
    sequence bracketed by a special "." token meaning start/end, e.g.
    "emma" becomes the bigrams:  .e  em  mm  ma  a.
"""

import torch


def read_names(path="names.txt"):
    return open(path).read().splitlines()


# ------------------------------------------------------------
# 1. Vocabulary
# ------------------------------------------------------------

def build_vocab(words):
    """Return (stoi, itos).

    stoi: dict mapping character -> integer index. Index 0 is reserved
    for the special "." boundary token (start AND end of every name).
    The other indices are the 26 lowercase letters, in sorted order.

    itos: the reverse mapping, integer -> character.

    Hint: sorted(set(''.join(words))) gives you the unique letters in
    a stable order. Build stoi as {'.': 0, then each letter: 1, 2, ...}.
    """

    stoi = { s:i for i,s in enumerate(['.'] + sorted(set(''.join(words)))) }
    itos = { i:s for s,i in stoi.items() }
    return stoi, itos


# ------------------------------------------------------------
# 2. The counting model
# ------------------------------------------------------------

def count_bigrams(words, stoi):
    """Return N: a (27, 27) torch.Tensor of INTEGER counts, where
    N[i, j] = how many times character j followed character i, across
    the whole dataset (with '.' bracketing every word).

    Hint: torch.zeros((27, 27), dtype=torch.int32) to start. For each
    word, build the bracketed character sequence ['.'] + list(word) +
    ['.'], then for each consecutive PAIR (ch1, ch2) in that sequence,
    increment N[stoi[ch1], stoi[ch2]] by 1.
    """

    N = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)
    for word in words:
        chars = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            N[stoi[ch1], stoi[ch2]] += 1
    return N


def counts_to_probs(N):
    """Convert the (27,27) count matrix N into a (27,27) probability
    matrix P, where each ROW sums to 1 (P[i] = probability distribution
    over "what comes after character i").

    Hint: N.float() first. Then divide each row by its own sum:
    row_sums = N.sum(dim=1, keepdim=True) — then P = N / row_sums.
    Add a small number (e.g. 1) to N before converting, so no
    probability is ever exactly zero ("model smoothing" — look up why
    a zero probability is catastrophic for the loss below).
    """

    P = (N + 1).float()
    row_sums = P.sum(dim=1, keepdim=True)
    P /= row_sums
    return P


# ------------------------------------------------------------
# 3. Sampling — generate names from P
# ------------------------------------------------------------

def sample_counting(P, itos, n=20, seed=2147483647):
    """Generate n names by sampling from P.

    Start at index stoi['.'] (== 0). Repeatedly sample the next
    character index from P[current_index] using torch.multinomial,
    append it, and stop when you sample index 0 again (the end token).
    Join the sampled characters (excluding the final '.') into a
    string. Return a list of n such generated names.

    Hint: torch.multinomial(P[ix], num_samples=1, generator=g).item()
    draws one index according to the probabilities in that row. Use a
    torch.Generator().manual_seed(seed) so results are reproducible.
    """

    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(n):
        name = []
        ix = 0  # start with the '.' token
        while True:
            ix = torch.multinomial(P[ix], num_samples=1, generator=g).item()
            if ix == 0:  # end token
                break
            name.append(itos[ix])
        names.append(''.join(name))
    return names


# ------------------------------------------------------------
# 4. Evaluate the counting model — negative log likelihood
# ------------------------------------------------------------

def nll_loss_counting(words, stoi, P):
    """Return the average negative log likelihood of the dataset under
    P — this is the "loss" for a model with no gradient descent at
    all, and is what you're trying to beat/match with training.

    For every bigram (ch1, ch2) in every word (bracketed with '.'):
        log_likelihood += log(P[stoi[ch1], stoi[ch2]])
        n += 1
    Return -(log_likelihood / n).

    Karpathy's reference value on the full dataset is approximately
    2.45 to 2.46. If you get something wildly different, check your
    smoothing in counts_to_probs and that P's rows actually sum to 1.
    """

    log_likelihood = 0.0
    n = 0
    for word in words:
        chars = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
            n += 1
    return -(log_likelihood / n).item()


# ------------------------------------------------------------
# 5. The SAME model, trained by gradient descent
# ------------------------------------------------------------

def build_training_set(words, stoi):
    """Return (xs, ys): two 1D torch.LongTensor of equal length, over
    every bigram (ch1, ch2) in the dataset (bracketed with '.').
    xs[k] = stoi[ch1], ys[k] = stoi[ch2] for the k-th bigram.

    This is just count_bigrams's data, unaggregated — one row per
    bigram occurrence instead of one row per (i,j) pair.
    """

    xs, ys = [], []
    for word in words:
        chars = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def train_gradient_descent(xs, ys, steps=100, lr=50.0, vocab_size=27, seed=2147483647):
    """Train a single (27, 27) weight matrix W by gradient descent so
    that softmax(one_hot(x) @ W) approximates P(next | current).

    Steps, each loop iteration:
      1. FORWARD:
         - one-hot encode xs into (N, 27) floats: F.one_hot(xs,
           num_classes=27).float()
         - logits = xenc @ W                      # (N, 27)
         - counts = logits.exp()                  # "fake counts"
         - probs = counts / counts.sum(1, keepdim=True)   # softmax
      2. LOSS:
         - loss = -probs[torch.arange(N), ys].log().mean()
         - (optionally add a tiny regularization term: + 0.01 *
           (W**2).mean(), to keep weights small — try both)
      3. BACKWARD:
         - W.grad = None
         - loss.backward()
      4. UPDATE:
         - W.data += -lr * W.grad

    Return the trained W and a list of the loss at every step.

    Hint on why softmax(one_hot @ W) is the SAME model as the counting
    one: exp(logits) plays the role of "counts," and normalizing by
    row sum plays the role of counts_to_probs. Gradient descent is
    finding the W whose implied counts match the data's actual counts.
    W must start with requires_grad=True — that's what makes .backward()
    track it.
    """
    import torch.nn.functional as F

    for k in range(steps):
        torch.manual_seed(seed + k)  # for reproducibility
        W = torch.randn((vocab_size, vocab_size), requires_grad=True)

        losses = []
        for step in range(steps):
            # FORWARD
            xenc = F.one_hot(xs, num_classes=vocab_size).float()
            logits = xenc @ W
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdim=True)

            # LOSS
            loss = -probs[torch.arange(len(xs)), ys].log().mean()
            loss += 0.01 * (W**2).mean()  # optional regularization

            # BACKWARD
            W.grad = None
            loss.backward()

            # UPDATE
            with torch.no_grad():
                W -= lr * W.grad

            losses.append(loss.item())

    return W, losses



# ------------------------------------------------------------
if __name__ == "__main__":
    words = read_names()
    stoi, itos = build_vocab(words)

    print("=" * 50)
    print("COUNTING MODEL")
    N = count_bigrams(words, stoi)
    P = counts_to_probs(N)
    loss = nll_loss_counting(words, stoi, P)
    print(f"counting model nll loss: {loss:.4f}  (Karpathy's reference: ~2.45-2.46)")
    print("sampled names:", sample_counting(P, itos, n=10))

    print("=" * 50)
    print("GRADIENT DESCENT MODEL (same model, different training)")
    xs, ys = build_training_set(words, stoi)
    W, losses = train_gradient_descent(xs, ys, steps=100)
    print(f"gradient descent final loss: {losses[-1]:.4f}")
    print("Do the two losses roughly match? They should — that's the whole point.")
