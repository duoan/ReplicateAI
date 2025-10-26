from typing import Optional, Tuple, Union

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
        d_key = key.size(-1)
        # (N, H, T, d_head) @ (N, H, d_head, T)
        # => (N, n_heads, T, T)
        scale = d_key**-0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -1e-9)
            else:
                scores = scores.masked_fill(mask == 0, -1e-9)

        attn_weights = F.softmax(
            scores, dim=-1
        )  # (N, H, T_query, T_key) very q weights cross keys
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
            attn_weights: (N, H, T, T), for visualization analysis, distillation, feature selection, pruning etc.
        """
        if key is None:  # self attention
            key = query
        if value is None:
            value = key

        N, T, _ = query.shape

        # Linear projections
        # (N, T, d_model) @ (d_model, d_model) => (N, T, d_model)
        query_projected = self.W_q(query)  # (N, T, d_model)
        key_projected = self.W_k(key)  # (N, T, d_model)
        value_projected = self.W_v(value)  # (N, T, d_model)

        # Split heads
        # (N, T, d_model) => (N, T, H, d_head)
        query = self._split_heads(query_projected)  # (N, T, H, d_head)
        key = self._split_heads(key_projected)  # (N, T, H, d_head)
        value = self._split_heads(value_projected)  # (N, T, H, d_head)

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
        activation="relu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # up sampling
        self.expansion = nn.Linear(d_model, d_ff, bias=bias)  # (d_model, d_ff)
        # down sampling
        self.reduction = nn.Linear(d_ff, d_model, bias=bias)  # (d_ff, d_model)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
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
        activation="relu",
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
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_attn=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src:    (N, T, d_model) - input embeddings or previous layer output
            src_mask: (N, 1, 1, T) or (N, H, T, T), optional attention mask

        Returns:
            output: (N, T, d_model)
        """
        # --- Self-Attention block ---
        attn_out, attn_weights = self.attn(src, mask=src_mask)
        attn_out = src + self.dropout(attn_out)  # residual connection
        attn_out = self.norm1(attn_out)  # Post-LN

        ff_out = self.ff(attn_out)
        out = attn_out + self.dropout(ff_out)  # residual connection
        out = self.norm2(out)  # Post-LN

        if return_attn:
            return out, attn_weights
        else:
            return out


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout_prob=0.1,
        bias=False,
        activation="relu",
        norm_eps=1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout_prob, bias)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout_prob, bias)
        self.ff = FeedForward(d_model, d_ff, dropout_prob, bias, activation)

        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=norm_eps)

        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)

    def forward(
        self,
        tgt: torch.Tensor,
        src: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        return_attn=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Pass the inputs (and mask) through the decoder layer.

        Args:

            tgt: the sequence to the decoder layer (required).
            src: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional). Shape: (N, 1, T_tgt, T_tgt)
            src_mask: the mask for the memory sequence (optional). Shape: (N, 1, 1, T_src)

        """
        # self attention for target
        tgt_self_attn_out, tgt_self_attn_weights = self.self_attn(tgt, mask=tgt_mask)
        tgt_self_attn_out = self.dropout1(tgt_self_attn_out)
        tgt_self_attn_out = self.norm1(tgt + tgt_self_attn_out)

        # cross attention with source
        # Query from target, Key and Value from source
        # src_mask shape (N, 1, 1, T_src) will broadcast to (N, H, T_tgt, T_src)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            query=tgt_self_attn_out, key=src, value=src, mask=src_mask
        )
        cross_attn_out = self.norm2(tgt_self_attn_out + self.dropout2(cross_attn_out))

        # feed forward
        ff_out = self.ff(cross_attn_out)
        out = self.norm3(cross_attn_out + self.dropout3(ff_out))

        if return_attn:
            return out, tgt_self_attn_weights, cross_attn_weights
        else:
            return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, n_position=5000):
        super().__init__()

        pe = torch.zeros(n_position, d_model)
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (N, T, d_model)
        T = x.size(1)
        # (N, T, d_model) + (1, T, d_model) ==> (N, T, d_model)
        pe = self.get_buffer("pe")
        return x + pe[:, :T]


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_id,
        tgt_pad_id,
        tgt_sos_id,
        n_src_vocab,
        n_tgt_vocab,
        n_positions,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout_prob=0.1,
        bias=False,
        activation="gelu",
        norm_eps=1e-6,
        n_encoder_layers=1,
        n_decoder_layers=1,
    ):
        super().__init__()

        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.tgt_sos_id = tgt_sos_id

        self.pos_enc = PositionalEncoding(d_model, n_positions)

        self.src_embed = nn.Embedding(n_src_vocab, d_model)  # input embedding
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, n_heads, d_ff, dropout_prob, bias, activation, norm_eps
                )
                for _ in range(n_encoder_layers)
            ]
        )

        self.tgt_embed = nn.Embedding(n_tgt_vocab, d_model)  # output embedding
        self.decoder = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model, n_heads, d_ff, dropout_prob, bias, activation, norm_eps
                )
                for _ in range(n_decoder_layers)
            ]
        )

        self.tgt_proj = nn.Linear(d_model, n_tgt_vocab, bias=bias)

    def _make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        # Shape: (N, 1, 1, T_src) - for masking padding in keys
        src_mask = (src != self.src_pad_id).unsqueeze(1).unsqueeze(2)
        return src_mask

    def _make_trg_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        N, tgt_len = tgt.shape
        
        # Padding mask: (N, 1, 1, T_tgt) - for masking padding in keys
        tgt_pad_mask = (tgt != self.tgt_pad_id).unsqueeze(1).unsqueeze(2)  # (N, 1, 1, T_tgt)
        
        # Causal mask: (1, 1, T_tgt, T_tgt)
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(0)
        
        # Combine: (N, 1, T_tgt, T_tgt)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask


    def forward(
        self,
        src_ids: torch.Tensor,  # (N, T_src)
        tgt_ids: torch.Tensor,  # (N, T_tgt)
        return_attn=False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, list[torch.Tensor]]]]:
        # Encoder
        src_mask = self._make_src_mask(src_ids)
        tgt_mask = self._make_trg_mask(tgt_ids)
        src_emb = self.src_embed(src_ids)  # （N, T_src, d_model
        src = self.pos_enc(src_emb)

        encoder_attn_maps = []
        for layer in self.encoder:
            if return_attn:
                src, attn = layer(src, src_mask, return_attn=True)
                encoder_attn_maps.append(attn)
            else:
                src = layer(src, src_mask)

        # Decoder
        tgt_emb = self.tgt_embed(tgt_ids)  # (N, T_tgt, d_model)
        tgt = self.pos_enc(tgt_emb)
        decoder_self_attn_maps, decoder_cross_attn_maps = [], []
        for layer in self.decoder:
            if return_attn:
                tgt, self_attn, cross_attn = layer(
                    tgt, src, src_mask=src_mask, tgt_mask=tgt_mask, return_attn=True
                )
                decoder_self_attn_maps.append(self_attn)
                decoder_cross_attn_maps.append(cross_attn)
            else:
                tgt = layer(tgt, src, src_mask=src_mask, tgt_mask=tgt_mask)

        # Logits
        logits = self.tgt_proj(tgt)  # (N, T_tgt, vocab)

        if return_attn:
            return logits, {
                "encoder": encoder_attn_maps,
                "decoder_self": decoder_self_attn_maps,
                "decoder_cross": decoder_cross_attn_maps,
            }
        else:
            return logits

    def greedy_decode(self, src, max_len=100, eos_id=None):
        self.eval()
        src_mask = self._make_src_mask(src)
        src_emb = self.pos_enc(self.src_embed(src))
        memory = src_emb
        for layer in self.encoder:
            memory = layer(memory, src_mask)

        N = src.size(0)
        ys = torch.full((N, 1), self.tgt_sos_id, dtype=torch.long, device=src.device)

        for i in range(max_len - 1):
            tgt_mask = self._make_trg_mask(ys)
            tgt_emb = self.pos_enc(self.tgt_embed(ys))
            out = tgt_emb
            for layer in self.decoder:
                out = layer(out, memory, src_mask=src_mask, tgt_mask=tgt_mask)
            logits = self.tgt_proj(out[:, -1, :])
            next_token = logits.argmax(dim=-1).unsqueeze(1)
            ys = torch.cat([ys, next_token], dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break
        return ys


def test_scaled_dot_product_attention_causal_only():
    attn = ScaledDotProductAttention()
    N, H, T, d = 2, 8, 10, 64
    Q = torch.randn(N, H, T, d)
    K = torch.randn(N, H, T, d)
    V = torch.randn(N, H, T, d)

    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=Q.device))
    out, w = attn(Q, K, V, mask=causal_mask)

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
