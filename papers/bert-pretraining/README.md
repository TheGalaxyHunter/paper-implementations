# BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019)

**Paper**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

**Authors**: Devlin, Chang, Lee, Toutanova (Google AI Language)

## Summary

BERT introduced a simple yet powerful idea: pre-train a deep bidirectional Transformer by
masking random tokens (Masked Language Model) and predicting whether two sentences are
consecutive (Next Sentence Prediction). This pre-trained model can then be fine-tuned with
just one additional output layer for a wide range of NLU tasks, achieving new state-of-the-art
results on eleven benchmarks.

## Key Contributions

- **Bidirectional pre-training**: Unlike GPT (left-to-right) or ELMo (shallow concatenation),
  BERT uses masked language modeling to enable deep bidirectional representations
- **Masked Language Model (MLM)**: Randomly masks 15% of tokens and trains the model to predict
  them, enabling bidirectional context without information leakage
- **Next Sentence Prediction (NSP)**: Binary classification task that helps the model understand
  sentence-pair relationships
- **Fine-tuning paradigm**: A single pre-trained model can be adapted to diverse downstream
  tasks with minimal architecture changes

## Implementation

| File | Description |
|------|-------------|
| `model.py` | BERT architecture: embeddings, encoder, MLM head, NSP head |
| `pretrain.py` | Pre-training with MLM and NSP objectives |
| `finetune.py` | Fine-tuning for sequence classification |
| `config.yaml` | Hyperparameters for BERT-base configuration |
| `notes.md` | Analysis of key design decisions |

## Running

### Pre-training (synthetic data for demonstration)

```bash
python pretrain.py
```

### Fine-tuning for classification

```bash
python finetune.py
```

## Architecture Overview

```
[CLS] token_1 token_2 ... [SEP] token_a token_b ... [SEP]
  |      |       |              |       |
Token Embeddings + Segment Embeddings + Position Embeddings
  |
[Transformer Encoder x 12]
  |
Hidden states for each token
  |
[CLS] -> NSP head (is_next / not_next)
[MASK] positions -> MLM head (predict original token)
```

## References

- Original paper: https://arxiv.org/abs/1810.04805
- Google's BERT repo: https://github.com/google-research/bert
