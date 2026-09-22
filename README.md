# makemore

Character-level name generators, built from scratch in PyTorch, on the same 32,033-name dataset. Growing collection — each model gets its own folder, added as it's built.

Follows Andrej Karpathy's ["makemore" lecture series](https://karpathy.ai/zero-to-hero.html).

## Models

| Folder | Model | Status |
|--------|-------|--------|
| [`bigram/`](bigram/) | Predicts the next character from only the current one, trained two ways | done |
| `mlp/` | Predicts from a window of previous characters, via a learned embedding | in progress |

More folders will be added as later parts of the series are built (RNN, WaveNet, and the eventual transformer version).

## bigram/

Given the current character, predict the next one — the simplest possible language model: 27×27 numbers (26 letters + one boundary token for start/end of a name).

Two training methods on the same data, compared directly:

1. **Counting.** Count how often each character follows each other character, normalize each row into a probability distribution. No gradient descent — pure statistics.
2. **Gradient descent.** The same model, reframed as one linear layer: `softmax(one_hot(x) @ W)`. `exp(logits)` plays the role of the counts, row-normalizing plays the role of the counting model's normalization. Trained with `.backward()`.

The point of comparing them: two completely different training procedures converge to nearly the same loss — evidence that gradient descent is finding the same distribution the counts already encode.

### Usage

```bash
pip install -r requirements.txt
cd bigram
python3 bigram.py
```

```python
from bigram import build_vocab, count_bigrams, counts_to_probs, sample_counting

words = open("../names.txt").read().splitlines()
stoi, itos = build_vocab(words)
N = count_bigrams(words, stoi)
P = counts_to_probs(N)
print(sample_counting(P, itos, n=10))
```

### Tests

```bash
python3 -m pytest bigram/tests/ -v
```

14 tests: vocabulary construction, bigram counting correctness (hand-verified on a small example), probability normalization, sampling stays within the known vocabulary, loss on a fully predictable toy dataset is near zero, and — the main integration test — the gradient-descent model converges close to the counting model's loss.

### Results

On the full dataset:

| Model | Loss |
|-------|------|
| Counting | 2.4544 |
| Gradient descent | 2.4907 |

Both land close together, as expected. Sample names generated from the counting model: `cexze`, `konimittain`, `llayn`, `ka`, `da`, `moliellavo` — recognizably name-shaped fragments, not real names, since the model only ever sees one character of context.

### Notes from building this

- A count matrix built with a hardcoded shape instead of deriving it from the vocabulary works by coincidence on the exact dataset it was written for, and breaks silently on any other one — the symptom can show up somewhere completely unrelated to the actual bug (a `KeyError` during sampling, caused by an oversized matrix built during counting).
- In-place tensor operations (`P /= row_sums`) on a tensor computed from itself a line earlier can raise at runtime; writing `P = P / row_sums` instead avoids it.
- A function with no `return` statement runs its full body and returns `None` — the resulting error appears at the call site, not at the missing line.

## Data

`names.txt`, shared across every model in this repo, one name per line.
