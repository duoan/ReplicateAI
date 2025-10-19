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
            query: (N, n_heads, T_q, d_k)
            key: (N, n_heads, T_k, d_k)
            value: (N, n_heads, T_k, d_v)
            mask: (N, 1, 1, T_k) or (N, 1, T_q, T_k)
                  0 for masked, 1 for valid positions.

        Returns:
            output: (N, n_heads, T_q, d_v)
            attention_weights: (N, n_heads, T_q, T_k)
        """

        d_k = query.size(-1)
        # (N, n_heads, T_q, d_k) @ (N, n_heads, d_k, T_k)
        # => (N, n_heads, T_q, T_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (N, n_heads, T_q, T_k) @ (N, n_heads, T_k, d_v)
        # => (N, n_heads, T_q, d_v)
        output = torch.matmul(attn_weights, value)

        return output, attn_weights


class MultiHeadAttention(nn.Module):

    def __init__(
            self,
            d_model=512,
            n_heads=8,
            dropout_prob=0.1,
            bias=True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"{d_model=} must be x times of {n_heads=}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_prob = dropout_prob
        self.d_kv = d_model // n_heads

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
        """Split last dim into (n_heads, d_k)"""
        N, T, _ = x.shape
        x = x.view(N, T, self.n_heads, self.d_kv)
        return x.transpose(1, 2)  # axis change

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (N, n_heads, T_q, T_k) into (N, T_q, n_heads, T_k)
        view
         (N, T_q, n_heads, T_k) => (N, T_q, n_heads * T_k)
        """
        N, H, T, d_k = x.shape
        # must using contiguous before view [geometry change]
        return x.transpose(1, 2).contiguous().view(N, T, H * d_k)

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (N, T_q, d_model)
            key:   (N, T_k, d_model) or None (defaults to self-attention)
            value: (N, T_k, d_model) or None
            mask:  (N, 1, 1, T_k) or (N, H, T_q, T_k), bool
        Returns:
            out: (N, T_q, d_model)
            attn_weights: (N, H, T_q, T_k)
        """
        if key is None:  # self attention
            key = query
        if value is None:
            value = key

        N, T_q, _ = query.shape

        # Linear projections
        # (N, T_q, d_model) @ (d_model, d_model) => (N, T_q, d_model)
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        # Split heads
        # (N, T_q, d_model) => ((N, T_q, n_heads, d_kv)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # Attention
        attn_out, attn_weights = self.attention(query, key, value, mask)
        # attn_out (N, n_heads, T_q, T_k)
        # attn_weights (N, n_heads, T_q, T_k)

        # Combine heads
        attn_out = self._combine_heads(attn_out)
        out = self.W_o(attn_out)
        out = self.dropout(out)
        return out, attn_weights


class PositionwiseFeedForward(nn.Module):

    def __init__(
            self,
            d_model=512,
            d_ff=2048,
            dropout_prob=0.1,
            bias=True,
            norm_eps=0.1,
            activation='relu',
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_prob = dropout_prob
        self.norm_eps = norm_eps
        self.W_q = nn.Linear(d_model, d_model, bias=bias)

        self.input_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.output_proj = nn.Linear(d_ff, d_model, bias=bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
           Args:
               x: (N, T, d_model)
           Returns:
               out: (N, T, d_model)
        """
        out = self.input_proj(x)  # (N, T, d_hidden)
        out = self.activation(out)  # nonlinearity
        out = self.dropout(out)  # regularization
        out = self.output_proj(out)  # (N, T, d_model)
        return out


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


def test_positionwise_feed_forward():
    ff = PositionwiseFeedForward(d_model=512, d_ff=2048, activation="gelu")
    x = torch.randn(2, 10, 512)
    out = ff(x)
    assert out.shape == (2, 10, 512)  # torch.Size([2, 10, 512])