from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Compute Scaled Dot-Product Attention
    (Vaswani et al., 2017, Eq. 1)
    """

    def __init__(self, dropout_prob: float = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (N, H, T, d_head)
            key: (N, H, T, d_head)
            value: (N, H, T, d_head)
            mask: (N, 1, 1, T) or (N, 1, T, T)
                  0 for masked, 1 for valid positions.

        Returns:
            output: (N, H, T, d_head)
            attention_weights: (N, H, T, T)
        """
        d_key = key.size(2)
        # (N, H, T, d_head) @ (N, H, d_head, T)
        # => (N, n_heads, T, T)
        scale = d_key ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # very q weights cross keys
        attn_weights = self.dropout(attn_weights)  # (N, H, T, T)

        # (N, H, T, T) @ (N, H, T, d_head)
        # => (N, H, T, d_head)
        output = torch.matmul(attn_weights, value)

        return output, attn_weights


class MultiHeadAttention(nn.Module):

    def __init__(
            self,
            d_model=512,
            n_heads=8,
            dropout_prob=0.1,
            bias=False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"{d_model=} must be x times of {n_heads=}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_prob = dropout_prob
        self.d_head = d_model // n_heads

        # projection layers for query, key, value
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)

        # Final output projection
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.attention = ScaledDotProductAttention(dropout_prob)
        # Normalization  + dropout
        self.dropout = nn.Dropout(dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(N, T, d_model) -> (N, T, H, d_head)"""
        N, T, _ = x.shape
        x = x.view(N, T, self.n_heads, self.d_head)
        return x.transpose(1, 2)  # axis change

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (N, H, T, d_head) transpose into (N, T, H, d_head)
        view
         (N, T, H, d_head) => (N, T, H * d_head) == (N, T, H, d_model)
        """
        N, H, T, d_head = x.shape
        # must using contiguous before view [geometry change]
        return x.transpose(1, 2).contiguous().view(N, T, H * d_head)

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (N, T, d_model)
            key:   (N, T, d_model) or None (defaults to self-attention)
            value: (N, T, d_model) or None
            mask:  (N, 1, 1, T) or (N, H, T, T), bool
        Returns:
            out: (N, T, d_model)
            attn_weights: (N, H, T, T)， for visualization analysis, distillation, feature selection, pruning etc.
        """
        if key is None:  # self attention
            key = query
        if value is None:
            value = key

        N, T, _ = query.shape

        # Linear projections
        # (N, T, d_model) @ (d_model, d_model) => (N, T, d_model)
        query = self.W_q(query)  # (N, T, d_model)
        key = self.W_k(key)  # (N, T, d_model)
        value = self.W_v(value)  # (N, T, d_model)

        # Split heads
        # (N, T, d_model) => (N, T, H, d_head)
        query = self._split_heads(query)  # (N, T, H, d_head)
        key = self._split_heads(key)  # (N, T, H, d_head)
        value = self._split_heads(value)  # (N, T, H, d_head)

        # Attention
        attn_out, attn_weights = self.attention(query, key, value, mask)
        # attn_out (N, H, T, d_head)
        # attn_weights (N, H, T, T)

        # Combine heads
        attn_out = self._combine_heads(attn_out)  # (N, T, d_model)
        out = self.W_o(attn_out)
        out = self.dropout(out)
        return out, attn_weights


class FeedForward(nn.Module):

    def __init__(
            self,
            d_model=512,
            d_ff=2048,
            dropout_prob=0.1,
            bias=True,
            activation='relu',
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # up sampling
        self.expansion = nn.Linear(d_model, d_ff, bias=bias)  # (d_model, d_ff)
        # down sampling
        self.reduction = nn.Linear(d_ff, d_model, bias=bias)  # (d_ff, d_model)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout_prob)  # regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
           Args:
               x: (N, T, d_model)
           Returns:
               out: (N, T, d_model)
        """
        out = self.expansion(x)  # (N, T, d_model) @ (d_model, d_ff) => (N, T, d_ff)
        out = self.activation(out)  # nonlinearity
        out = self.dropout(out)  # regularization
        out = self.reduction(out)  # (N, T, d_ff) @ (d_ff, d_model) => (N, T, d_model)
        return out


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer (Pre-LayerNorm variant)
    -------------------------------------------------
    Structure:
        x -> LN -> MHA -> Dropout -> Add
          -> LN -> FFN -> Dropout -> Add

    Args:
        d_model: model dimension
        n_heads: number of attention heads
        d_ff: feed-forward hidden dimension (expansion size)
        dropout_prob: dropout probability
    """

    def __init__(
            self,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            dropout_prob=0.1,
            bias=False,
            activation='relu',
            norm_eps=1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.attn = MultiHeadAttention(d_model, n_heads, dropout_prob, bias)
        self.ff = FeedForward(d_model, d_ff, dropout_prob, bias, activation)

        # LayerNorm (post)
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (N, T, d_model) - input embeddings or previous layer output
            mask: (N, 1, 1, T) or (N, H, T, T), optional attention mask

        Returns:
            output: (N, T, d_model)
        """
        # --- Self-Attention block ---
        attn_out, attn_weights = self.attn(x, mask=mask)
        x = x + self.dropout(attn_out)  # residual connection
        x = self.norm1(x)  # Post-LN

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)  # residual connection
        x = self.norm2(x)  # Post-LN

        return x


def test_scaled_dot_product_attention_causal_only():
    attn = ScaledDotProductAttention()
    N, H, T, d = 2, 8, 10, 64
    Q = torch.randn(N, H, T, d)
    K = torch.randn(N, H, T, d)
    V = torch.randn(N, H, T, d)

    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=Q.device), diagonal=1)
    out, w = attn(Q, K, V, mask=causal)

    assert out.shape == (N, H, T, d)
    assert w.shape == (N, H, T, T)


def test_multi_head_attention():
    mha = MultiHeadAttention(d_model=512, n_heads=8)
    X = torch.randn(2, 10, 512)  # (N, T, d_model)
    out, w = mha(X)
    print(out.shape, w.shape)
    assert out.shape == (2, 10, 512)
    assert w.shape == (2, 8, 10, 10)


def test_feed_forward():
    ff = FeedForward(d_model=512, d_ff=2048, activation="gelu")
    x = torch.randn(2, 10, 512)
    out = ff(x)
    assert out.shape == (2, 10, 512)  # torch.Size([2, 10, 512])
