from dataclasses import dataclass
from typing import Optional, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

"""
https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
"""


@dataclass
class ViTConfig:
    num_classes: int
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = hidden_size * 4
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    qkv_bias: bool = False
    encoder_stride: int = 14
    pooler_output_size: Optional[int] = None
    pooler_act = "tanh"
    attention_impl: Literal["eager", "flash_attention_2", "flash_attention_2"] = "eager"


class ViTPatchEmbedding(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.projector = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
        )
        assert config.image_size % config.patch_size == 0

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.flatten = nn.Flatten(start_dim=2)  # Flatten the spatial dimensions

    def forward(self, pixel_values: Tensor) -> Tensor:
        # conv2, H = W = (224 - 16 + 2 * 0) / 16 + 1 = 14
        out = self.projector(pixel_values)  # (N, out_channels, H', W') => (N, 768, 14, 14)
        out = self.flatten(out)  # (N, out_channels, H'*W') => (N, 767, 196)
        return out.transpose(1, 2)  # Transpose to get shape (H, H'*W', out_channels)


class ViTEmbedding(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbedding(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.path_size = config.patch_size
        self.config = config

    def forward(
            self,
            pixel_values: Tensor,
            bool_masked_pos: Optional[torch.BoolTensor] = None,
    ) -> Tensor:
        batch_size, num_channels, height, width = pixel_values.shape

        embeddings = self.patch_embeddings(pixel_values)  # (batch_size, num_patches, hidden_size)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (1,1, hidden_size) => (batch_size, 1, hidden_size)
        # cat (batch_size, 1, hidden_size) (batch_size, seq_length, hidden_size) => (batch_size, num_patches + 1, hidden_size)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # (batch_size, num_patches + 1, hidden_size) + (1, num_patches + 1, hidden_size) [broadcast]
        # => (batch_size, num_patches + 1, hidden_size)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class ViTAttention(nn.Module):
    """
    layer norm + multi-head self-attention + drop + residual
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention_impl = config.attention_impl
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attn_drop = nn.Dropout(config.hidden_dropout_prob)
        self.scale: float = self.attention_head_size ** -0.5

        # pre-norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # qkv, projectors (hidden_size, all_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, return_attention=False) -> Union[Tensor, tuple[Tensor, Tensor]]:
        # hidden_state: (batch_size, num_patches + 1, hidden_size)
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size

        # (batch_size, num_patches + 1, hidden_size) @ (hidden_size, all_head_size)=> (batch_size, num_patches + 1, all_head_size)
        # view      => (batch_size, num_patches + 1, num_attention_heads, attention_head_size)
        # transpose => (batch_size, num_attention_heads, num_patches + 1, attention_head_size)
        query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)

        if self.attention_impl == "eager":
            # (batch_size, num_attention_heads, num_patches + 1, attention_head_size)
            # @ (batch_size, num_attention_heads, attention_head_size, num_patches + 1)
            # => (batch_size, num_attention_heads, num_patches + 1, num_patches + 1)
            attn = torch.matmul(query_layer, key_layer.transpose(-2, -1)) / self.scale
            attn = attn.softmax(dim=-1)

            # (batch_size, num_attention_heads, num_patches + 1, num_patches + 1)
            # @ (batch_size, num_attention_heads, num_patches + 1, attention_head_size)
            # => (batch_size, num_attention_heads, num_patches + 1, attention_head_size)
            # transpose =>  (batch_size, num_patches + 1, num_attention_heads, attention_head_size)
            # view => (batch_size, num_patches + 1, num_attention_heads * attention_head_size)
            context = (
                # (batch_size, num_attention_heads, num_patches + 1, attention_head_size)
                torch.matmul(self.attn_drop(attn), value_layer)
                .transpose(1, 2)  # (batch_size, num_patches + 1, num_attention_heads, attention_head_size)
                .contiguous()
                .view(batch_size, -1, self.all_head_size)
                # (batch_size, num_patches + 1, num_attention_heads * attention_head_size)
            )
        elif self.attention_impl == "sdpa":
            context = F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=None,
                is_causal=False,
                scale=self.scale,
                dropout_p=self.config.dropout_p,
            )
        elif self.attention_impl == "flash_attention_2":
            from flash_attn import flash_attn_func
            context = flash_attn_func(
                query_layer,
                key_layer,
                value_layer,
                dropout_p=self.config.dropout_p,
                softmax_scale=self.scale,
            )
        else:
            raise NotImplementedError("not implemented")

        # output
        output = self.out_proj(context)
        output = self.out_drop(output) + hidden_states

        if return_attention:
            return output, attn
        else:
            return output


class ViTFeedForward(nn.Module):
    """
    norm + MLP + residual
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        y = self.norm(hidden_states)
        return self.net(y) + hidden_states


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(config.num_hidden_layers):
            self.layers.append(
                nn.Sequential(
                    ViTAttention(config),
                    ViTFeedForward(config),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.embedding = ViTEmbedding(config)
        self.encoder = ViTEncoder(config)
        self.mlp_head = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, pixel_values: Tensor, labels: Optional[Tensor] = None) -> dict[str, Tensor]:
        x = self.embedding(pixel_values)
        x = self.encoder(x)
        cls = x[:, 0]  # CLS token
        logits = self.mlp_head(cls)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {
            "logits": logits,
            "loss": loss,
        }
