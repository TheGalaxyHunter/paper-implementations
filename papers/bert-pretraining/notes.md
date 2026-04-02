# Notes: BERT Pre-training

## Bidirectional Context vs. GPT

GPT uses a left-to-right (autoregressive) language model: each token can only attend to
previous tokens. This is natural for generation but limits the model's ability to capture
context from both directions during pre-training.

BERT's key insight is that for understanding tasks (classification, NER, QA), you want
the representation of each token to be informed by both left and right context. The
challenge is: how do you train bidirectionally without the model "seeing the answer"?

## Masked Language Model (MLM)

The solution is masking. BERT randomly selects 15% of tokens for prediction. Of those:

- 80% are replaced with `[MASK]`
- 10% are replaced with a random token
- 10% are kept unchanged

Why not always use `[MASK]`? Because `[MASK]` never appears during fine-tuning, creating
a mismatch between pre-training and fine-tuning. The random replacement and identity cases
force the model to maintain good representations for all tokens, not just masked ones.

The 15% masking rate is a compromise. Higher rates give more signal per sequence but also
make each prediction harder (less context). Lower rates waste computation on unmasked
tokens that contribute no loss.

## MLM vs. Autoregressive Pre-training

A common criticism of MLM is that it's "less efficient" than autoregressive modeling because
only 15% of tokens produce a training signal per example (vs. 100% for GPT). This is true,
and it means BERT needs more training data/steps to converge.

However, the bidirectional context appears to more than compensate for this. BERT-base
outperforms GPT-1 on most benchmarks despite being a similar size, suggesting the quality
of each training signal matters more than the quantity.

## Next Sentence Prediction (NSP)

NSP trains the model to predict whether sentence B follows sentence A in the original text.
50% of the time it does (label: IsNext), 50% it's a random sentence (label: NotNext).

The motivation is that many tasks (QA, NLI) require understanding relationships between
sentence pairs. The `[CLS]` token's representation after pre-training captures this
relationship.

Later work (RoBERTa, ALBERT) questioned whether NSP is actually helpful. RoBERTa showed
that removing NSP and using longer sequences performs better. Still, it's an important
part of the original BERT design.

## Fine-tuning Paradigm

BERT's fine-tuning approach is remarkably simple:

1. Take the pre-trained model
2. Add a single task-specific output layer
3. Fine-tune all parameters end-to-end

For classification, use the `[CLS]` token's final hidden state. For token-level tasks
(NER), use each token's hidden state. For span tasks (QA), add start/end classifiers.

This "pre-train then fine-tune" paradigm fundamentally changed NLP. Before BERT, most
tasks required carefully designed architectures. After BERT, the same base model works
for nearly everything.

## Embedding Design

BERT uses three types of embeddings, summed together:

1. **Token embeddings**: Standard vocabulary embeddings (WordPiece tokenization)
2. **Segment embeddings**: Distinguish sentence A from sentence B (only two segments)
3. **Position embeddings**: Learned (not sinusoidal like the original Transformer)

The use of learned position embeddings (vs. sinusoidal in the original Transformer) is
a practical choice. Since BERT has a fixed maximum sequence length (512), learned
embeddings work fine and may capture position-specific patterns better.

## Implementation Observations

- The `[CLS]` token is always prepended. Its final hidden state serves as the aggregate
  sequence representation for classification tasks. This is a simple but effective pooling
  strategy.
- Layer normalization and GELU activation (instead of ReLU) are used throughout. GELU
  provides smoother gradients and appears to help with pre-training stability.
- BERT-base: 12 layers, 768 hidden, 12 heads, 110M parameters.
  BERT-large: 24 layers, 1024 hidden, 16 heads, 340M parameters.
- Pre-training used BooksCorpus (800M words) + English Wikipedia (2,500M words).
  Training took 4 days on 16 TPU chips for BERT-base.
