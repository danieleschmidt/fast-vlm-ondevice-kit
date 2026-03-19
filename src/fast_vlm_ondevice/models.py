"""
Core VLM model components for on-device inference.

All models are designed for edge deployment:
  - Minimal parameter count
  - CPU-friendly operations
  - No pretrained weights required (architecture demo)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    Lightweight 3-layer CNN for visual feature extraction.

    Architecture:
        Conv2d(3→32, k=3) → BN → ReLU → MaxPool
        Conv2d(32→64, k=3) → BN → ReLU → MaxPool
        Conv2d(64→128, k=3) → BN → ReLU → AdaptiveAvgPool
        Linear → embedding_dim

    Outputs a fixed-size embedding regardless of input spatial resolution.
    """

    def __init__(self, embedding_dim: int = 128, img_size: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv_layers = nn.Sequential(
            # Layer 1: 3 → 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size/2

            # Layer 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size/4

            # Layer 3: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # fixed 4x4 spatial output
        )

        # Project 128 (per spatial token) → embedding_dim
        # Applied per-token so both forward() and encode_as_tokens() share it
        self.proj = nn.Linear(128, embedding_dim)

    def encode_as_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return projected spatial feature tokens.
        Used by CrossModalAttention.

        Returns:
            (B, 16, embedding_dim) — 16 spatial tokens of dim embedding_dim
        """
        features = self.conv_layers(x)                          # (B, 128, 4, 4)
        tokens = features.flatten(2).permute(0, 2, 1)          # (B, 16, 128)
        return self.proj(tokens)                                # (B, 16, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — RGB image batch

        Returns:
            (B, embedding_dim) — visual embedding (mean-pooled tokens)
        """
        tokens = self.encode_as_tokens(x)          # (B, 16, embedding_dim)
        return tokens.mean(dim=1)                  # (B, embedding_dim)


class TextEncoder(nn.Module):
    """
    Small transformer encoder for text queries.

    Architecture:
        Token embedding (vocab_size → d_model)
        Positional encoding
        2× TransformerEncoderLayer (d_model=64, nhead=4, ffn=128)
        Mean pooling → d_model embedding
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm is more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, token_ids: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (B, seq_len) — integer token ids
            mask: (B, seq_len) bool mask — True = ignore (padding)

        Returns:
            (B, d_model) — text embedding (mean-pooled)
        """
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0)  # (1, L)

        x = self.token_embed(token_ids) + self.pos_embed(positions)  # (B, L, d_model)
        x = self.transformer(x, src_key_padding_mask=mask)            # (B, L, d_model)
        x = self.norm(x)

        # Mean pooling (ignore masked positions)
        if mask is not None:
            keep = (~mask).float().unsqueeze(-1)                      # (B, L, 1)
            x = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)

        return x  # (B, d_model)

    def encode_as_tokens(
        self, token_ids: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return full token sequence for cross-attention.

        Returns:
            (B, seq_len, d_model)
        """
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.norm(x)


class CrossModalAttention(nn.Module):
    """
    Cross-attention: visual tokens (query) attend to text tokens (key/value).

    This lets image regions focus on relevant text concepts.
    Output: refined visual tokens aligned to the text query.
    """

    def __init__(self, visual_dim: int = 128, text_dim: int = 64, num_heads: int = 4):
        super().__init__()
        # Project both modalities to a shared attention dimension
        attn_dim = max(visual_dim, text_dim)
        # Round to multiple of num_heads
        attn_dim = math.ceil(attn_dim / num_heads) * num_heads

        self.visual_proj = nn.Linear(visual_dim, attn_dim)
        self.text_proj = nn.Linear(text_dim, attn_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, batch_first=True
        )
        self.out_proj = nn.Linear(attn_dim, visual_dim)
        self.norm = nn.LayerNorm(visual_dim)

    def forward(
        self, visual_tokens: torch.Tensor, text_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: (B, num_visual, visual_dim)
            text_tokens:   (B, seq_len, text_dim)

        Returns:
            (B, num_visual, visual_dim) — attention-refined visual tokens
        """
        q = self.visual_proj(visual_tokens)   # (B, nv, attn_dim)
        k = self.text_proj(text_tokens)        # (B, nt, attn_dim)
        v = k

        attended, _ = self.attn(q, k, v)      # (B, nv, attn_dim)
        out = self.out_proj(attended)          # (B, nv, visual_dim)

        # Residual + norm
        return self.norm(visual_tokens + out)  # (B, nv, visual_dim)


class TinyVLM(nn.Module):
    """
    Full on-device Vision-Language Model.

    Pipeline:
        image → ImageEncoder.encode_as_tokens → visual_tokens (B, 16, 128)
        text  → TextEncoder.encode_as_tokens  → text_tokens   (B, L, 64)
        CrossModalAttention(visual_tokens, text_tokens)
        → mean pool visual → image_emb (B, 128)
        → text mean pool   → text_emb  (B, 64)
        → concat & MLP     → matching_score (B,)

    Returns a scalar similarity score for each (image, text) pair.
    """

    def __init__(
        self,
        visual_dim: int = 128,
        text_dim: int = 64,
        vocab_size: int = 256,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(embedding_dim=visual_dim)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            d_model=text_dim,
            max_seq_len=max_seq_len,
        )
        self.cross_attn = CrossModalAttention(
            visual_dim=visual_dim, text_dim=text_dim
        )

        # Matching head: concat both embeddings → scalar score
        self.match_head = nn.Sequential(
            nn.Linear(visual_dim + text_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            images:    (B, 3, H, W)
            token_ids: (B, seq_len)
            text_mask: (B, seq_len) bool, True = ignore padding

        Returns:
            (B,) matching scores (logits)
        """
        visual_tokens = self.image_encoder.encode_as_tokens(images)       # (B, 16, 128)
        text_tokens = self.text_encoder.encode_as_tokens(token_ids, text_mask)  # (B, L, 64)

        refined_visual = self.cross_attn(visual_tokens, text_tokens)      # (B, 16, 128)

        image_emb = refined_visual.mean(dim=1)                            # (B, 128)
        text_emb = text_tokens.mean(dim=1)                                # (B, 64)

        combined = torch.cat([image_emb, text_emb], dim=-1)               # (B, 192)
        score = self.match_head(combined).squeeze(-1)                     # (B,)
        return score
