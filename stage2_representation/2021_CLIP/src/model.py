import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np


def tokenize_text(text: str) -> torch.Tensor:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-2")
    print(enc.encode(text))
    print(enc.max_token_value)
    return None


class CLIP(nn.Module):
    def __init__(self, vocab_size: int = 49408, pad_id: int = 0, seq_len: int = 77,
                 num_text_layers: int = 4, num_text_heads: int = 8,
                 text_ffn_hidden_size: int = 512, text_dropout: float = 0.1):
        super().__init__()
        image_feature_hidden_size = 128
        text_feature_hidden_size = 128
        align_feature_hidden_size = 512
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # --- Image encoder backbone (tiny CNN) ---
        self.image_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=1)
        )
        self.image_proj = nn.Linear(image_feature_hidden_size, align_feature_hidden_size, bias=False)

        # --- Text encoder (token + position embedding + TransformerEncoder + masked mean pool) ---
        self.pad_id = pad_id
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, text_feature_hidden_size, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(seq_len, text_feature_hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_feature_hidden_size,
            nhead=num_text_heads,
            dim_feedforward=text_ffn_hidden_size,
            dropout=text_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_text_layers)
        self.text_proj = nn.Linear(text_feature_hidden_size, align_feature_hidden_size, bias=False)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W] -> features: [B, align_feature_hidden_size]
        x = self.image_backbone(images)           # [B, 128, 1, 1]
        x = x.flatten(start_dim=1)                # [B, 128]
        x = self.image_proj(x)                    # [B, align_dim]
        return x

    def encode_texts(self, texts: torch.Tensor) -> torch.Tensor:
        # texts: [B, T] token ids -> features: [B, align_feature_hidden_size]
        embeddings = self.token_embedding(texts)  # [B, T, text_hidden]
        seq_len = embeddings.size(1)
        positions = torch.arange(seq_len, device=texts.device).unsqueeze(0)
        embeddings = embeddings + self.position_embedding(positions)  # [B, T, text_hidden]
        # True values will be ignored by the transformer
        key_padding_mask = (texts == self.pad_id)  # [B, T], bool
        embeddings = self.text_transformer(embeddings, src_key_padding_mask=key_padding_mask)
        mask = (texts != self.pad_id).float()     # [B, T]
        masked_sum = (embeddings * mask.unsqueeze(-1)).sum(dim=1)  # [B, text_hidden]
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)      # [B, 1]
        pooled = masked_sum / denom                                # [B, text_hidden]
        x = self.text_proj(pooled)                                 # [B, align_dim]
        return x

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calculate similarity between images and texts
        Args:
            images (torch.Tensor): images tensor with shape [batch_size, channels, height, width]
            texts (torch.Tensor): texts tensor with shape [batch_size, seq_len]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: logits for images and texts
        """
        image_features = self.encode_images(images)  # (batch_size, align_feature_hidden_size)
        text_features = self.encode_texts(texts)     # (batch_size, align_feature_hidden_size)

        # normalize L2
        eps = 1e-6
        image_features = image_features / torch.norm(image_features, p=2, dim=-1, keepdim=True).clamp(min=eps)
        text_features = text_features / torch.norm(text_features, p=2, dim=-1, keepdim=True).clamp(min=eps)

        # calculate pairwise similarity
        logits_per_image =  self.logit_scale.exp() * image_features @ text_features.T # (batch_size, batch_size)
        logits_per_text =  logits_per_image.t()

        # (batch_size, batch_size)
        return logits_per_image, logits_per_text


if __name__ == '__main__':
    tokenize_text("hello world")

    model = CLIP(vocab_size=49408, pad_id=0, seq_len=77)
    
    images = torch.randn(4, 3, 224, 224)
    texts = torch.randint(0, 49408, (4, 77))
    logits_per_image, logits_per_text = model(images, texts)
    print(logits_per_image)
    print(logits_per_text)