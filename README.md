# makemore

Character-level name generators, built from scratch in PyTorch, on the same 32,033-name dataset. Growing collection — each model gets its own folder, added as it's built.

Follows Andrej Karpathy's ["makemore" lecture series](https://karpathy.ai/zero-to-hero.html).

## Models

| Folder | Model | Status |
|--------|-------|--------|
| [`bigram/`](bigram/) | Predicts the next character from only the current one, trained two ways | done |
| [`mlp/mlp.py`](mlp/mlp.py) | Predicts from a window of 10 previous characters, via a learned embedding | done — dev loss 2.0872 |
| [`mlp/mlp_batchnorm.py`](mlp/mlp_batchnorm.py) | The same MLP with batch normalization | done — dev loss 2.0423 |

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

Predicts the next character from the previous **10** characters. Each character is looked up in a learned 10-dimensional embedding table, the 10 embeddings are concatenated into one 100-number vector, and a small network (a `tanh` hidden layer of 200 units, then a linear layer over the 27 characters) scores the next one. 25,897 parameters, 80/10/10 train/dev/test split.

Training: minibatch gradient descent, batch size 400, 20,000 steps, learning rate 0.1 for the first 10,000 steps and 0.01 after. `W1` and `W2` start as 0.1 × a standard normal.

### Usage

```bash
python3 mlp/mlp.py                       # about 25 seconds
python3 -m pytest mlp/tests/ -v
```

Output:

```
dataset : 32,033 names, vocab 27
examples: train 182,625 | dev 22,655 | test 22,866
params  : 25,897
----------------------------------------
step      0/20000 | loss 3.9453
step   2000/20000 | loss 2.1488
step   4000/20000 | loss 2.1730
step   6000/20000 | loss 2.1540
step   8000/20000 | loss 2.1027
step  10000/20000 | loss 2.0169
step  12000/20000 | loss 2.0286
step  14000/20000 | loss 2.0064
step  16000/20000 | loss 2.0661
step  18000/20000 | loss 2.1175
step  19999/20000 | loss 2.0275
----------------------------------------
train loss: 2.0459
dev loss  : 2.0872
test loss : 2.0876
samples: ['dexber', 'jaliera', 'carityn', 'mellimanna', 'nylany', 'zakiah', 'samiyah', 'javer', 'gotal', 'molie']
```

The per-step loss bounces around because each step sees only 400 examples; the train, dev and test numbers at the end are measured on the full splits. Train and dev are close (2.05 vs 2.09), so the model is not overfitting much. Across 3 seeds the dev loss is 2.0872, 2.0877 and 2.0889 (mean 2.0879), so run-to-run variation is under 0.002.

### Result

Dev loss **2.0872**, test loss 2.0876, down from 2.1480 in the first version (context window 10, batch 50, `W` scaled by 0.01, constant learning rate 0.1). That version was already below the roughly 2.17 of the lecture, which used a context of 3.

**Context window.** The first big gain came from widening the context. At batch 32 with a constant learning rate (single runs):

| Context window | Batch | Dev loss |
|---:|---:|---:|
| 3 | 32 | 2.2457 |
| 10 | 32 | 2.1675 |
| 10 | 50 | 2.1480 |

**Batch size, weight scale and learning-rate schedule.** Three settings changed between the first version and the current one. Dev loss for all eight combinations (context 10, 20,000 steps, a single run each). The current settings vary by under 0.002 across seeds, so differences of 0.01 or more are meaningful and smaller ones are noise:

| Batch | `W` scale | Constant lr 0.1 | lr 0.1 → 0.01 at step 10,000 |
|---:|---:|---:|---:|
| 50 | 0.01 | 2.1480 (first version) | 2.1460 |
| 50 | 0.1 | 2.1393 | 2.1090 |
| 400 | 0.01 | 2.0799 | 2.1160 |
| 400 | 0.1 | **2.0706** | 2.0872 (current) |

- **Batch size 50 → 400 helped most**: 0.02 to 0.07 depending on the other two settings, and about 0.07 with a constant learning rate.
- **`W` scale 0.01 → 0.1 helped a little** with a constant learning rate (about 0.01) and more once the rate drops (0.03 to 0.04).
- **The learning-rate drop is not a clear win.** At batch 50 with `W` at 0.1 it helps by 0.03. At batch 400 it makes things worse by 0.02 to 0.04: 10,000 steps at the high rate are not enough, and the model is still improving when the rate drops. The current settings keep the drop and score 2.0872; the same settings with a constant rate score 2.0706.

### Errors hit along the way

1. **`TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`** during the update step, with the warning *"The .grad attribute of a Tensor that is not a leaf Tensor is being accessed"*. The weights were created as `torch.randn(..., requires_grad=True) * 0.01`. The multiplication produces a new tensor with a `grad_fn`, so it is not a leaf and `.backward()` never fills its `.grad`. Fix: scale first, then call `requires_grad_(True)` on every parameter.
2. **`TypeError: 'NoneType' object is not subscriptable`** in the test for decreasing loss. `train()` ran its whole loop but never built or returned the list of losses, so the caller received `None`. Fix: collect `loss.item()` each step and `return losses`. The same missing-`return` mistake showed up in `bigram/`.
3. **Loss stuck above 2.2 with no error.** With the weights left unscaled (`torch.randn` with no factor), the starting loss was 33 instead of about 3.3, and `test_forward_loss_near_uniform_at_init` failed. Adding steps looked like progress (dev 2.2480 at 80,000 steps, 2.2047 at 200,000) but it was only the slow-starting model catching up. Scaling the weights fixed it.
4. **Too many steps.** With the scale restored and a constant learning rate, dev loss was 2.0564 at 80,000 steps and 2.0823 at 200,000 while train loss kept falling to 1.85 (context 8, batch 400): the model starts memorizing the training names. Dev loss is best at roughly 40,000 to 80,000 steps.

The first round of fixes was not in the saved file when the tests were rerun, so the same error appeared a second time. Check that the file on disk actually contains the change before rerunning.

## mlp/ with batch normalization

`mlp/mlp_batchnorm.py` is the same network with batch normalization between the first linear layer and the `tanh`: each hidden unit is normalized across the batch, then given a learned scale (`bngain`) and shift (`bnbias`). There is no `b1`, because subtracting the batch mean right afterwards would cancel it. Running estimates of the mean and variance (momentum 0.001) are used at evaluation time, when a single example has no batch to normalize against. 26,097 learnable parameters; the running statistics are not learned. Training settings are the same as `mlp.py` (batch 400, 20,000 steps, learning rate 0.1 → 0.01 at step 10,000), with `W1` and `W2` at 0.01 × a standard normal.

### Usage

```bash
python3 mlp/mlp_batchnorm.py             # about 35 seconds
python3 -m pytest mlp/tests/ -v          # mlp and batchnorm tests together
```

Output:

```
dataset : 32,033 names, vocab 27
examples: train 182,625 | dev 22,655 | test 22,866
params  : 26,097 learnable
----------------------------------------
step      0/20000 | loss 3.3064
step   2000/20000 | loss 2.2624
step   4000/20000 | loss 2.1010
step   6000/20000 | loss 2.0381
step   8000/20000 | loss 2.0029
step  10000/20000 | loss 1.9640
step  12000/20000 | loss 1.9754
step  14000/20000 | loss 1.9931
step  16000/20000 | loss 1.9975
step  18000/20000 | loss 1.8532
step  19999/20000 | loss 1.8922
----------------------------------------
train loss: 1.9627
dev loss  : 2.0423
test loss : 2.0396
```

### Result

Dev loss **2.0423**, test loss 2.0396, against 2.0872 and 2.0876 for the plain MLP. Dev loss over 3 seeds:

| Seed | MLP | MLP + batchnorm |
|---|---:|---:|
| default | 2.0872 | 2.0423 |
| 1 | 2.0877 | 2.0462 |
| 2 | 2.0889 | 2.0465 |
| mean | 2.0879 | 2.0450 |

Batchnorm is better on every seed, by about 0.04, while the seeds themselves vary by 0.002 (MLP) and 0.004 (batchnorm). The comparison is not purely batchnorm against none, though: the two files also differ in the scale of `W1`/`W2` (0.01 here, 0.1 in `mlp.py`) and in how the biases start (zeros for `b2` here, a standard normal in `mlp.py`).

Weight scale still matters with batchnorm. At batch 50 with a constant learning rate, 3-seed mean dev loss was 2.1312 with `W1` at 0.01 and 2.1463 with a Kaiming-style scale (`5/3 / sqrt(fan_in)`), so the larger initialization did not help.

### Errors hit along the way

1. **`TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`** in the update step. A loop that called `requires_grad_(True)` on every entry of `params` also switched it on for `bnmean_running` and `bnvar_running`. Those two are updated by a running average and never receive a gradient, so their `.grad` stayed `None`, and the check `if p.requires_grad` in the training loop no longer skipped them. Fix: enable gradients only on the six learnable tensors (`C`, `W1`, `W2`, `b2`, `bngain`, `bnbias`).
2. **The same error again**, from `b2 = torch.zeros(..., requires_grad=True) * 0`. Multiplying a tensor that already tracks gradients creates a non-leaf, so its `.grad` is never filled. Fix: create `b2` with plain `torch.zeros` and enable gradients afterwards. This is the same trap as the weights in `mlp.py`; `test_learnable_params_are_trainable_leaves` now checks for it.

## Data

`names.txt`, shared across every model in this repo, one name per line.
