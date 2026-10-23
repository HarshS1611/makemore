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
python3 bigram/bigram.py   # about 3 seconds
```

Output:

```
counting model loss:         2.4544
samples: ['cexze', 'momasurailezitynn', 'konimittain', 'llayn', 'ka', 'da', 'staiyaubrtthrigotai', 'moliellavo', 'ke', 'teda']
gradient descent final loss: 2.4901
```

Using it from Python (run inside `bigram/`):

```python
from bigram import read_names, build_vocab, count_bigrams, counts_to_probs, sample_counting

words = read_names()
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
| Gradient descent | 2.4901 |

Both land close together, as expected. Sample names generated from the counting model: `cexze`, `konimittain`, `llayn`, `ka`, `da`, `moliellavo` — recognizably name-shaped fragments, not real names, since the model only ever sees one character of context.

### Notes from building this

- **Hardcoded shape.** Building the count matrix as a literal `(27, 27)` instead of `(len(stoi), len(stoi))` works on this dataset by coincidence and breaks on any other vocabulary. The errors showed up far from the cause: a size-mismatch error when checking row sums, and a `KeyError` during sampling when the oversized matrix produced an index outside the vocabulary.
- **Missing `return`.** A function that does all its work but never returns gives back `None`. The error appears at the call site (`cannot unpack NoneType`), not at the missing line.
- **A stray outer loop.** An extra `for` around the training loop re-initialized the weights every iteration: 100 separate 100-step runs, keeping only the last. The tests still passed, because the result was correct, just 100× too slow. Removing it took the full script from 3 min 43 s to about 3 seconds. Passing tests don't prove the code is efficient.

## Data

`names.txt`, shared across every model in this repo, one name per line.
