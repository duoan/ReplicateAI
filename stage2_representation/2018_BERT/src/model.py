from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor


@dataclass
class BertConfig:
    vocab_size: int
    hidden_size: int
    seq_length: int
    type_vocab_size: int
    layer_norm_eps: float = 1e-12
    dropout_prob: float = 0.0
    num_attention_heads: int = 8


def scaled_dot_product_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        training: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Scaled dot product attention between queries and keys.
    Args:
        query (Tensor): query tensor, shape (batch_size, num_heads, query_dim, d_head)
        key (Tensor): key tensor, shape (batch_size, num_heads, key_dim, d_head)
        value (Tensor): value tensor, shape (batch_size, num_heads, value_dim, d_head)
        attn_mask (Tensor): attn_mask tensor, shape (batch_size, num_heads)
            - boolean mask (True = mask out) of shape broadcastable to [B, H, query_dim, key_dim],
        key_padding_mask (optional): boolean of shape [B, key_dim] (True = pad to mask out).
        dropout_p (float, optional): dropout probability applied to attention weights.
        training (bool, optional): set True to enable dropout.
    Returns:
        Tuple[Tensor, Tensor]: scaled dot product attention tensor and attention wights tensor
    """
    d_k = query.size(-1)
    scale = 1 / d_k ** -0.5
    # (batch_size, num_heads, query_dim, d_head) @ (batch_size, num_heads, d_head, key_dim)
    # => (batch_size, num_heads, query_dim, key_dim)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask, float('-inf'))

    if key_padding_mask is not None:
        kpm = key_padding_mask.view(query.size(0), 1, 1, key.size(2))
        scores = scores.masked_fill(kpm, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)  # compute the softmax on keys for every query

    # (batch_size, num_heads, query_dim, key_dim) @ (batch_size, num_heads, value_dim, d_head)
    # key_dim = value_dim
    # => (batch_size, num_heads, query_dim, d_head)
    if dropout_p > 0.0 and training:
        attention_output = torch.matmul(F.dropout(attention_weights, p=dropout_p), value)
    else:
        attention_output = torch.matmul(attention_weights, value)

    return attention_output, attention_weights


def make_causal_mask(query_dim, key_dim, device=None) -> Tensor:
    """
    Lower-triangular (causal) boolean mask of shape [1, 1, query_dim, key_dim] (True = masked).
    """
    mask = torch.ones([query_dim, key_dim], dtype=torch.bool, device=device)
    # Zero out the lower triangle (including diagonal)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (B=1, H=1, query_dim, key_dim)


class BertEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embeddings = nn.Embedding(config.seq_length, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
            persistent=False
        )

    def forward(
            self,
            input_ids: Tensor,  # (B, L)
            token_type_ids: Optional[Tensor] = None,  # (B, L)
            position_ids: Optional[Tensor] = None,  # (B, L)
    ) -> Tensor:
        B, L = input_ids.shape
        if position_ids is None:
            position_ids = self.position_ids[:, :L]  # (1, L)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        w = self.word_embeddings(input_ids)  # (B, L, H)
        p = self.pos_embeddings(position_ids)  # (1, L, H) -> broadcast
        s = self.token_type_embeddings(token_type_ids)  # (B, L, H)

        x = w + p + s
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size % num_attention_heads != 0")

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = 1 / self.head_dim ** 5  # 1/sqrt(d_k)

        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.k = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout_prob)

    def _split_heads(self, x: Tensor) -> Tensor:
        B, L, H = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: Tensor) -> Tensor:
        # B, num_heads, L, head_dim
        B, num_heads, L, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, L, num_heads * head_dim)

    def forward(
            self,
            hidden_states: Tensor,  # (B, L, H)
            attention_mask: Optional[Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B, L, _ = hidden_states.shape

        # (B,L,H) * (H, H) => (B, L, H) => split into => (B, num_heads, L, head_dim)
        q = self._split_heads(self.q(hidden_states))
        k = self._split_heads(self.k(hidden_states))
        v = self._split_heads(self.v(hidden_states))

        # (B, num_heads, L, head_dim) @ (B, num_heads, head_dim, L)
        # => (B, num_heads, L, L)
        scores = torch.bmm(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(self.dropout(attn), v)  # [B, num_heads, L, head_dim]
        ctx = self._merge_head(attn)

        return (ctx, attn) if output_attentions else (ctx, None)


def run_scaled_dot_product_attention():
    torch.manual_seed(0)
    hidden_dim = 64
    num_heads = 2
    seq_length = 32
    head_dim = hidden_dim // num_heads
    q = torch.rand((1, num_heads, seq_length, head_dim), dtype=torch.float32)
    k = torch.rand((1, num_heads, seq_length, head_dim), dtype=torch.float32)
    v = torch.rand((1, num_heads, seq_length, head_dim), dtype=torch.float32)
    attn_mask = make_causal_mask(seq_length, seq_length)

    output, weights = scaled_dot_product_attention(q, k, v, attn_mask)

    print(output.size())
    print(weights.size())

    assert output.size() == torch.Size((1, 2, 32, 32))
    assert weights.size() == torch.Size((1, 2, 32, 32))

    F.scaled_dot_product_attention(q, k, v)


if __name__ == '__main__':
    run_scaled_dot_product_attention()
