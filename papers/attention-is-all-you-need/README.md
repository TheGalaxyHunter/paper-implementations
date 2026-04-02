# Attention Is All You Need (Vaswani et al., 2017)

**Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**Authors**: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin

## Summary

This paper introduced the Transformer architecture, which replaces recurrent layers entirely
with multi-head self-attention mechanisms. The key insight is that attention alone, without
any recurrence or convolution, is sufficient for modeling sequence-to-sequence tasks. This
enables massive parallelization during training and achieves state-of-the-art results on
machine translation benchmarks.

## Key Contributions

- **Scaled dot-product attention**: Efficient attention mechanism that scales queries and keys
  by `1/sqrt(d_k)` to prevent vanishing gradients in softmax
- **Multi-head attention**: Multiple parallel attention heads allow the model to attend to
  information from different representation subspaces at different positions
- **Positional encoding**: Sinusoidal functions inject position information without learned
  parameters, enabling generalization to longer sequences than seen during training
- **Encoder-decoder architecture**: Six stacked layers each for encoder and decoder, with
  cross-attention connecting them

## Implementation

| File | Description |
|------|-------------|
| `model.py` | Full Transformer: multi-head attention, positional encoding, encoder/decoder stacks |
| `train.py` | Training loop with learning rate warmup and label smoothing |
| `config.yaml` | Hyperparameters matching the paper's base model configuration |
| `notes.md` | Personal notes on key insights and design decisions |

## Running

```bash
python train.py
```

The training script uses a synthetic copy task by default for quick validation.
Modify `config.yaml` to point to real translation data for full training.

## Architecture Overview

```
Input Embeddings + Positional Encoding
            |
    [Encoder x N]
     - Multi-Head Self-Attention
     - Feed-Forward Network
     - Layer Norm + Residual
            |
    [Decoder x N]
     - Masked Multi-Head Self-Attention
     - Multi-Head Cross-Attention
     - Feed-Forward Network
     - Layer Norm + Residual
            |
    Linear + Softmax
```

## References

- Original paper: https://arxiv.org/abs/1706.03762
- "The Annotated Transformer" (Harvard NLP): https://nlp.seas.harvard.edu/2018/04/03/attention.html
