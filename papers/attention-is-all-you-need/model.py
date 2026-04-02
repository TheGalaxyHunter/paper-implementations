"""
Transformer: Attention Is All You Need (Vaswani et al., 2017)
https://arxiv.org/abs/1706.03762

Full implementation of the Transformer architecture from scratch.
Each component maps directly to the paper's description in Section 3.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention (Section 3.2.1).

    Computes attention as:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    The scaling factor 1/sqrt(d_k) prevents the dot products from growing
    large in magnitude, which would push softmax into regions with extremely
    small gradients.
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            query: (batch, n_heads, seq_len_q, d_k)
            key:   (batch, n_heads, seq_len_k, d_k)
            value: (batch, n_heads, seq_len_k, d_v)
            mask:  Broadcastable mask. Positions with True are masked (ignored).

        Returns:
            output:  (batch, n_heads, seq_len_q, d_v)
            weights: (batch, n_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)

        # QK^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = torch.matmul(weights, value)
        return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (Section 3.2.2).

    Instead of performing a single attention function with d_model-dimensional
    keys, values, and queries, we project them h times with different learned
    projections to d_k, d_k, and d_v dimensions. This allows the model to
    jointly attend to information from different representation subspaces.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Learned linear projections for Q, K, V, and output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask:  Optional mask tensor

        Returns:
            (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # Linear projections, then reshape to (batch, n_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, _ = self.attention(q, k, v, mask=mask)

        # Concatenate heads and apply output projection
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(attn_output)


class PositionWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network (Section 3.3).

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    Applied independently to each position. The inner layer has dimension d_ff
    (typically 2048), while input and output have dimension d_model (512).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (Section 3.5).

    Injects position information using sine and cosine functions of different
    frequencies:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This allows the model to learn relative positions, since for any fixed
    offset k, PE(pos+k) can be represented as a linear function of PE(pos).
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer (Section 3.1).

    Each layer has two sub-layers:
    1. Multi-head self-attention
    2. Position-wise feed-forward network

    Both sub-layers use residual connections followed by layer normalization:
        output = LayerNorm(x + Sublayer(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x:        (batch, seq_len, d_model)
            src_mask: Optional source mask

        Returns:
            (batch, seq_len, d_model)
        """
        # Sub-layer 1: Multi-head self-attention
        attn_output = self.self_attention(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Sub-layer 2: Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer (Section 3.1).

    Each layer has three sub-layers:
    1. Masked multi-head self-attention (prevents attending to future positions)
    2. Multi-head cross-attention over encoder output
    3. Position-wise feed-forward network

    All sub-layers use residual connections + layer normalization.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:              (batch, tgt_seq_len, d_model)
            encoder_output: (batch, src_seq_len, d_model)
            src_mask:       Mask for encoder output (padding mask)
            tgt_mask:       Mask for decoder input (causal + padding mask)

        Returns:
            (batch, tgt_seq_len, d_model)
        """
        # Sub-layer 1: Masked self-attention
        attn_output = self.self_attention(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Sub-layer 2: Cross-attention over encoder output
        cross_output = self.cross_attention(x, encoder_output, encoder_output, mask=src_mask)
        x = self.norm2(x + self.dropout2(cross_output))

        # Sub-layer 3: Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """Stack of N encoder layers."""

    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """Stack of N decoder layers."""

    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """Full Transformer Model (Vaswani et al., 2017).

    Combines token embeddings, positional encoding, encoder stack, decoder stack,
    and output linear projection. Supports weight tying between embedding and
    output layers as described in the paper.

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model:        Model dimension (default: 512)
        n_heads:        Number of attention heads (default: 8)
        n_encoder_layers: Number of encoder layers (default: 6)
        n_decoder_layers: Number of decoder layers (default: 6)
        d_ff:           Feed-forward inner dimension (default: 2048)
        dropout:        Dropout rate (default: 0.1)
        max_seq_len:    Maximum sequence length for positional encoding
        tie_weights:    Whether to tie embedding and output projection weights
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Embeddings (Section 3.4)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder and decoder stacks
        self.encoder = TransformerEncoder(n_encoder_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(n_decoder_layers, d_model, n_heads, d_ff, dropout)

        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying: share weights between target embedding and output projection
        if tie_weights and src_vocab_size == tgt_vocab_size:
            self.output_projection.weight = self.tgt_embedding.weight

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize parameters using Xavier uniform, as is standard practice."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """Encode source sequence.

        Args:
            src:      (batch, src_seq_len) token indices
            src_mask: Optional padding mask

        Returns:
            (batch, src_seq_len, d_model) encoder output
        """
        # Scale embeddings by sqrt(d_model) as described in Section 3.4
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return self.encoder(x, mask=src_mask)

    def decode(
        self,
        tgt: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        """Decode target sequence given encoder output.

        Args:
            tgt:            (batch, tgt_seq_len) token indices
            encoder_output: (batch, src_seq_len, d_model)
            src_mask:       Optional source padding mask
            tgt_mask:       Optional target causal mask

        Returns:
            (batch, tgt_seq_len, d_model) decoder output
        """
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return self.decoder(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        """Full forward pass: encode source, decode target, project to vocabulary.

        Args:
            src:      (batch, src_seq_len) source token indices
            tgt:      (batch, tgt_seq_len) target token indices
            src_mask: Optional source padding mask
            tgt_mask: Optional target causal + padding mask

        Returns:
            (batch, tgt_seq_len, tgt_vocab_size) logits over target vocabulary
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)

    @staticmethod
    def generate_causal_mask(size: int, device: torch.device | None = None) -> Tensor:
        """Generate a causal (upper-triangular) mask for the decoder.

        Prevents positions from attending to subsequent positions.
        Returns a boolean mask where True means "mask this position".

        Args:
            size:   Sequence length
            device: Target device

        Returns:
            (1, 1, size, size) boolean mask
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def generate_padding_mask(x: Tensor, pad_idx: int = 0) -> Tensor:
        """Generate a padding mask from token indices.

        Args:
            x:       (batch, seq_len) token indices
            pad_idx: Index of the padding token

        Returns:
            (batch, 1, 1, seq_len) boolean mask
        """
        return (x == pad_idx).unsqueeze(1).unsqueeze(2)
