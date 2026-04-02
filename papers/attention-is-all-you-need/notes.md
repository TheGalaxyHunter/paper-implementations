# Notes: Attention Is All You Need

## Why Replace Recurrence with Attention?

RNNs process tokens sequentially, which creates two problems:

1. **Training bottleneck**: Sequential computation prevents parallelization across time steps.
   A sequence of length `n` requires `O(n)` sequential operations.
2. **Long-range dependencies**: Despite LSTMs and GRUs, information still degrades over long
   distances. The path length between distant positions grows linearly.

Self-attention connects every position to every other position in `O(1)` path length and
`O(n^2 * d)` total computation, which is actually faster than RNNs for typical sequence
lengths (`n < d`).

## Scaled Dot-Product Attention

The attention function is:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

The scaling factor `1/sqrt(d_k)` is critical. Without it, for large `d_k`, the dot products
grow large in magnitude, pushing softmax into regions with extremely small gradients. This
is a subtle but important detail: additive attention doesn't have this problem, but
dot-product attention is faster and more space-efficient, so the scaling fix is worthwhile.

## Multi-Head Attention

Instead of performing a single attention function with `d_model`-dimensional keys, values,
and queries, the paper projects them `h` times with different learned linear projections to
`d_k`, `d_v`, and `d_k` dimensions respectively.

This lets the model jointly attend to information from different representation subspaces at
different positions. With a single attention head, averaging would inhibit this.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

## Positional Encoding Choices

The paper uses sinusoidal positional encodings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Why sinusoidal and not learned?

1. **Extrapolation**: Sinusoidal encodings can generalize to sequence lengths longer than
   those seen during training. Learned embeddings cannot.
2. **Relative positions**: For any fixed offset `k`, `PE(pos+k)` can be represented as a
   linear function of `PE(pos)`, which may allow the model to learn relative positioning.
3. **Results are similar**: The paper notes that learned positional embeddings produced
   nearly identical results, so the choice is somewhat aesthetic.

## Residual Connections and Layer Normalization

Every sub-layer (attention, feed-forward) has a residual connection followed by layer
normalization:

```
output = LayerNorm(x + Sublayer(x))
```

This is critical for training deep networks. The paper uses 6 layers for both encoder and
decoder. Without residual connections, training such deep attention networks would be
extremely difficult.

## The Feed-Forward Network

Each layer contains a simple two-layer FFN:

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

This is applied position-wise (independently to each position). The inner dimension is 2048
while the model dimension is 512. This expansion-compression pattern appears throughout
modern architectures.

## Masking in the Decoder

The decoder uses masked self-attention to prevent positions from attending to subsequent
positions. This is implemented by setting future positions to `-inf` before the softmax,
ensuring the prediction for position `i` depends only on known outputs at positions less
than `i`.

## Implementation Observations

- Weight tying between the embedding layers and the pre-softmax linear transformation
  improves performance and reduces parameters.
- Label smoothing (epsilon = 0.1) hurts perplexity but improves accuracy and BLEU score.
- The learning rate schedule (warmup + inverse square root decay) is important for stable
  training.
- Dropout is applied to attention weights and after each sub-layer, which is essential for
  regularization in this architecture.
