# makemore

Character-level name generators, built from scratch in PyTorch, on the same 32,033-name dataset. Growing collection — each model gets its own folder, added as it's built.

Follows Andrej Karpathy's ["makemore" lecture series](https://karpathy.ai/zero-to-hero.html).

## Models

| Folder | Model | Status |
|--------|-------|--------|
| [`bigram/`](bigram/) | Predicts the next character from only the current one, trained two ways | done |
| [`mlp/`](mlp/) | Predicts from a window of 10 previous characters, via a learned embedding | done — dev loss 2.1480 |
| `mlp_batchnorm/` | The same MLP with batch normalization | in progress |

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

## mlp/

Predicts the next character from the previous **10** characters. Each character is looked up in a learned 10-dimensional embedding table, the 10 embeddings are concatenated into one 100-number vector, and a small network (`tanh` hidden layer of 200 units, then a linear layer over the 27 characters) scores the next one. 25,897 parameters. Trained with minibatch gradient descent on a 80/10/10 train/dev/test split.

### Usage

```bash
python3 mlp/mlp.py          # about 5 seconds
python3 -m pytest mlp/tests/ -v
```

Output:

```
dataset : 32,033 names, vocab 27
examples: train 182,625 | dev 22,655 | test 22,866
params  : 25,897
----------------------------------------
step      0/20000 | loss 3.5444
step   2000/20000 | loss 2.4600
step   4000/20000 | loss 2.0496
step   6000/20000 | loss 1.9673
step   8000/20000 | loss 2.2350
step  10000/20000 | loss 2.2046
step  12000/20000 | loss 2.0200
step  14000/20000 | loss 2.0526
step  16000/20000 | loss 2.1553
step  18000/20000 | loss 2.1524
step  19999/20000 | loss 1.9111
----------------------------------------
train loss: 2.1243
dev loss  : 2.1480
samples: ['dexber', 'jaleura', 'carvin', 'kelline', 'train', 'luwan', 'kaida', 'samiyah', 'javer', 'gotti']
```

The per-step loss bounces around (2.0 → 2.2 → 1.9) because each step sees only 50 examples; the train and dev numbers at the end are measured on the full splits. Train and dev are close (2.12 vs 2.15), so the model is not overfitting.

### Result

Dev loss **2.1480**, below the roughly 2.17 that the lecture's version reaches. The difference comes from two changes to the lecture's setup, measured one at a time (same seed, 20,000 steps, constant learning rate 0.1):

| Context window | Batch size | Train loss | Dev loss |
|---:|---:|---:|---:|
| 3 | 32 | 2.2339 | 2.2457 |
| 3 | 50 | 2.2220 | 2.2333 |
| 10 | 32 | 2.1458 | 2.1675 |
| 10 | 50 | 2.1243 | **2.1480** |

- **A longer context window is what mattered.** Going from 3 to 10 characters improved dev loss by about 0.08. Run-to-run variation across 3 seeds, measured at a context window of 3, was about 0.007, so this is well outside noise.
- **A larger batch helped only a little** (about 0.01–0.02), which is close to the noise level, so treat it as probably real but small.
- Each row is a single run, so the small differences are less certain than the large one.

Not everything helped. At a context window of 3, a smaller bias initialization (dev 2.2401) or dropping the learning rate late in training (2.2202–2.2364) each moved dev loss by under 0.03 on their own.

### Errors hit along the way

1. **`TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`** during the update step, with the warning *"The .grad attribute of a Tensor that is not a leaf Tensor is being accessed"*. The weights were created as `torch.randn(..., requires_grad=True) * 0.01`. The multiplication produces a new tensor with a `grad_fn`, so it is not a leaf and `.backward()` never fills its `.grad`. Fix: scale first, then call `requires_grad_(True)` on every parameter.
2. **`TypeError: 'NoneType' object is not subscriptable`** in the test for decreasing loss. `train()` ran its whole loop but never built or returned the list of losses, so the caller received `None`. Fix: collect `loss.item()` each step and `return losses`. This is the same missing-`return` mistake that showed up in `bigram/`.
3. **Loss stuck at about 2.24, above the reference.** Nothing was broken, so no error message: with a context window of 3, small initial weights and a constant learning rate, the model simply plateaus higher. Widening the window to 10 fixed it.

The first round of fixes was not in the saved file when the tests were rerun, so the same error appeared a second time. Check that the file on disk actually contains the change before rerunning.

## Data

`names.txt`, shared across every model in this repo, one name per line.
