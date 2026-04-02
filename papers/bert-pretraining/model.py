"""
BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019)
https://arxiv.org/abs/1810.04805

Full implementation of the BERT architecture from scratch, including:
- Token, segment, and position embeddings
- Transformer encoder stack
- Masked Language Model (MLM) head
- Next Sentence Prediction (NSP) head
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def gelu(x: Tensor) -> Tensor:
    """Gaussian Error Linear Unit activation (Hendrycks & Gimpel, 2016).

    BERT uses GELU instead of ReLU. It provides smoother gradients and
    appears to help with pre-training stability.

    GELU(x) = x * Phi(x), where Phi is the CDF of the standard normal.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertEmbeddings(nn.Module):
    """BERT Input Embeddings (Section 3.2 of the paper, Figure 2).

    Combines three embedding types:
    1. Token embeddings: WordPiece vocabulary embeddings
    2. Segment embeddings: Distinguish sentence A (0) from sentence B (1)
    3. Position embeddings: Learned positional encodings (unlike the original Transformer)

    The sum of these three is passed through LayerNorm and dropout.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        input_ids: Tensor,
        segment_ids: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            input_ids:  (batch, seq_len) token indices
            segment_ids: (batch, seq_len) segment indices (0 or 1)

        Returns:
            (batch, seq_len, hidden_size)
        """
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        embeddings = (
            self.token_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.segment_embeddings(segment_ids)
        )

        return self.dropout(self.layer_norm(embeddings))


class BertSelfAttention(nn.Module):
    """Multi-head self-attention for BERT.

    Same mechanism as the original Transformer, but BERT only uses the encoder
    (no causal masking by default). Padding positions are masked to prevent
    attention to padding tokens.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0

        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            hidden_states:  (batch, seq_len, hidden_size)
            attention_mask: (batch, 1, 1, seq_len) - 0 for real tokens, large negative for padding

        Returns:
            (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V and reshape for multi-head attention
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_size)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_size)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_size)

        # (batch, num_heads, seq_len, head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(weights, v)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.output(context)


class BertLayer(nn.Module):
    """Single BERT encoder layer.

    Each layer consists of:
    1. Self-attention with residual connection and layer norm
    2. Intermediate (expansion) feed-forward with GELU
    3. Output (compression) feed-forward with residual connection and layer norm
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = BertSelfAttention(hidden_size, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.attention_dropout = nn.Dropout(p=dropout)

        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.output_dropout = nn.Dropout(p=dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            hidden_states:  (batch, seq_len, hidden_size)
            attention_mask: Optional padding mask

        Returns:
            (batch, seq_len, hidden_size)
        """
        # Self-attention block
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.attention_norm(hidden_states + self.attention_dropout(attn_output))

        # Feed-forward block
        intermediate_output = gelu(self.intermediate(hidden_states))
        layer_output = self.output_dense(intermediate_output)
        hidden_states = self.output_norm(hidden_states + self.output_dropout(layer_output))

        return hidden_states


class BertEncoder(nn.Module):
    """Stack of BERT encoder layers."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [BertLayer(hidden_size, num_heads, intermediate_size, dropout) for _ in range(num_layers)]
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertPooler(nn.Module):
    """Pool the [CLS] token's hidden state for classification tasks.

    Takes the first token's representation, passes it through a dense layer
    and tanh activation. This pooled output is used for NSP during pre-training
    and for classification during fine-tuning.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            (batch, hidden_size) - pooled representation from [CLS] token
        """
        cls_token = hidden_states[:, 0]
        return self.activation(self.dense(cls_token))


class MLMHead(nn.Module):
    """Masked Language Model prediction head.

    Transforms hidden states back to vocabulary logits for predicting
    masked tokens. Uses a dense layer + GELU + LayerNorm before the
    output projection (which shares weights with the token embeddings).
    """

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) or subset of positions
        Returns:
            (batch, seq_len, vocab_size) logits
        """
        x = gelu(self.dense(hidden_states))
        x = self.layer_norm(x)
        return self.decoder(x)


class NSPHead(nn.Module):
    """Next Sentence Prediction head.

    Binary classifier that predicts whether sentence B follows sentence A
    in the original document.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, pooled_output: Tensor) -> Tensor:
        """
        Args:
            pooled_output: (batch, hidden_size) from BertPooler
        Returns:
            (batch, 2) logits for [IsNext, NotNext]
        """
        return self.classifier(pooled_output)


class Bert(nn.Module):
    """BERT: Bidirectional Encoder Representations from Transformers.

    Core BERT model without task-specific heads. Returns the full sequence
    of hidden states and the pooled [CLS] representation.

    Args:
        vocab_size:              Size of the WordPiece vocabulary
        hidden_size:             Dimension of hidden representations
        num_hidden_layers:       Number of Transformer encoder layers
        num_attention_heads:     Number of attention heads per layer
        intermediate_size:       Dimension of the feed-forward expansion layer
        max_position_embeddings: Maximum sequence length
        type_vocab_size:         Number of segment types (default: 2)
        dropout:                 Dropout rate
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout
        )
        self.encoder = BertEncoder(
            num_hidden_layers, hidden_size, num_attention_heads, intermediate_size, dropout
        )
        self.pooler = BertPooler(hidden_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following the BERT paper's convention."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        segment_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            input_ids:      (batch, seq_len) token indices
            segment_ids:    (batch, seq_len) segment indices (0 or 1)
            attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding

        Returns:
            sequence_output: (batch, seq_len, hidden_size) all token representations
            pooled_output:   (batch, hidden_size) [CLS] token representation
        """
        # Convert attention mask to additive mask for attention scores
        if attention_mask is not None:
            # (batch, seq_len) -> (batch, 1, 1, seq_len)
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask.float()) * -10000.0
        else:
            extended_mask = None

        embeddings = self.embeddings(input_ids, segment_ids)
        sequence_output = self.encoder(embeddings, extended_mask)
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output


class BertForPreTraining(nn.Module):
    """BERT model with MLM and NSP heads for pre-training.

    Combines the base BERT model with both pre-training objectives:
    1. Masked Language Model: Predict masked tokens
    2. Next Sentence Prediction: Predict if sentence B follows A
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.bert = Bert(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
            type_vocab_size=config["type_vocab_size"],
            dropout=config["hidden_dropout"],
        )
        self.mlm_head = MLMHead(config["hidden_size"], config["vocab_size"])
        self.nsp_head = NSPHead(config["hidden_size"])

        # Tie MLM output weights with token embeddings
        self.mlm_head.decoder.weight = self.bert.embeddings.token_embeddings.weight

    def forward(
        self,
        input_ids: Tensor,
        segment_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            mlm_logits: (batch, seq_len, vocab_size)
            nsp_logits: (batch, 2)
        """
        sequence_output, pooled_output = self.bert(input_ids, segment_ids, attention_mask)
        mlm_logits = self.mlm_head(sequence_output)
        nsp_logits = self.nsp_head(pooled_output)
        return mlm_logits, nsp_logits


class BertForSequenceClassification(nn.Module):
    """BERT model with a classification head for fine-tuning.

    Uses the pooled [CLS] representation with a simple linear classifier
    on top. This is the standard approach for tasks like sentiment analysis,
    NLI, and sentence-pair classification.
    """

    def __init__(self, config: dict, num_classes: int) -> None:
        super().__init__()
        self.bert = Bert(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
            type_vocab_size=config["type_vocab_size"],
            dropout=config["hidden_dropout"],
        )
        self.dropout = nn.Dropout(p=config["hidden_dropout"])
        self.classifier = nn.Linear(config["hidden_size"], num_classes)

    def forward(
        self,
        input_ids: Tensor,
        segment_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Returns:
            (batch, num_classes) classification logits
        """
        _, pooled_output = self.bert(input_ids, segment_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)
