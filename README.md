# bigram-language-model

A character-level bigram language model, trained two different ways on 32,032 names, in PyTorch.

## How it works

Given the current character, predict the next one. This is the simplest possible language model: only 27×27 numbers (26 letters + one boundary token marking start/end of a name), small enough to inspect by hand.

Two training methods, on the same data, compared directly:

1. **Counting.** Count how often each character follows each other character, then normalize each row into a probability distribution. No gradient descent, no training loop — pure statistics.
2. **Gradient descent.** The same model, reframed as a single linear layer: `softmax(one_hot(x) @ W)`. `exp(logits)` plays the role of the counts, and row-normalizing plays the role of the counting model's normalization. Trained with `.backward()` and plain gradient descent.

The point of comparing them: two completely different training procedures converge to nearly the same loss, which is direct evidence that gradient descent is finding the same distribution the counts already encode.

## Usage

```bash
pip install -r requirements.txt
python3 bigram.py
```

```python
from bigram import build_vocab, count_bigrams, counts_to_probs, sample_counting

words = open("names.txt").read().splitlines()
stoi, itos = build_vocab(words)
N = count_bigrams(words, stoi)
P = counts_to_probs(N)
print(sample_counting(P, itos, n=10))
```

## Tests

```bash
python3 -m pytest tests/ -v
```

14 tests: vocabulary construction, bigram counting correctness (hand-verified counts on a small example), probability normalization, sampling stays within the known vocabulary, loss on a fully predictable toy dataset is near zero, and — the main integration test — the gradient-descent model converges close to the counting model's loss.

## Results

On the full 32,032-name dataset:

| Model | Loss |
|-------|------|
| Counting | 2.4544 |
| Gradient descent | 2.4907 |

Both land close together, as expected. Sample names generated from the counting model: `cexze`, `konimittain`, `llayn`, `ka`, `da`, `moliellavo` — recognizably name-shaped fragments, not real names, since the model only ever sees one character of context.

## Notes from building this

- A count matrix built with a hardcoded shape instead of deriving it from the vocabulary works by coincidence on the exact dataset it was written for, and breaks silently on any other one — the symptom can show up somewhere completely unrelated to the actual bug (a `KeyError` during sampling, caused by an oversized matrix built during counting).
- In-place tensor operations (`P /= row_sums`) on a tensor computed from itself a line earlier can raise at runtime; writing `P = P / row_sums` instead avoids it.
- A function with no `return` statement runs its full body and returns `None` — the resulting error appears at the call site, not at the missing line.

## Reference

Follows Andrej Karpathy's "The spelled-out intro to language modeling: building makemore" (part 1).
