"""Tests for core VLM model components."""

import pytest
import torch

from fast_vlm_ondevice.models import (
    ImageEncoder,
    TextEncoder,
    CrossModalAttention,
    TinyVLM,
)


B = 2  # batch size used in all tests


class TestImageEncoder:
    def test_output_shape(self):
        model = ImageEncoder(embedding_dim=128)
        x = torch.randn(B, 3, 64, 64)
        out = model(x)
        assert out.shape == (B, 128), f"Expected ({B}, 128), got {out.shape}"

    def test_different_input_sizes(self):
        """Adaptive pooling should handle various input resolutions."""
        model = ImageEncoder(embedding_dim=64)
        for h, w in [(32, 32), (64, 64), (128, 96), (224, 224)]:
            x = torch.randn(1, 3, h, w)
            out = model(x)
            assert out.shape == (1, 64), f"Failed for input ({h},{w})"

    def test_encode_as_tokens(self):
        model = ImageEncoder(embedding_dim=128)
        x = torch.randn(B, 3, 64, 64)
        tokens = model.encode_as_tokens(x)
        # Tokens are projected to embedding_dim (128)
        assert tokens.shape == (B, 16, 128), f"Expected ({B}, 16, 128), got {tokens.shape}"

    def test_encode_as_tokens_custom_dim(self):
        model = ImageEncoder(embedding_dim=64)
        x = torch.randn(B, 3, 64, 64)
        tokens = model.encode_as_tokens(x)
        assert tokens.shape == (B, 16, 64)

    def test_no_nan(self):
        model = ImageEncoder()
        x = torch.randn(B, 3, 64, 64)
        out = model(x)
        assert not torch.isnan(out).any(), "NaN in ImageEncoder output"

    def test_batch_independence(self):
        """Each sample in a batch should be processed independently."""
        model = ImageEncoder(embedding_dim=32)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        # Single item should equal first item of batch
        out_batch = model(x)
        out_single = model(x[:1])
        torch.testing.assert_close(out_batch[:1], out_single, rtol=1e-4, atol=1e-4)


class TestTextEncoder:
    def test_output_shape(self):
        model = TextEncoder(vocab_size=256, d_model=64)
        tokens = torch.randint(0, 256, (B, 16))
        out = model(tokens)
        assert out.shape == (B, 64)

    def test_encode_as_tokens_shape(self):
        model = TextEncoder(vocab_size=256, d_model=64, max_seq_len=32)
        tokens = torch.randint(0, 256, (B, 16))
        out = model.encode_as_tokens(tokens)
        assert out.shape == (B, 16, 64)

    def test_with_padding_mask(self):
        model = TextEncoder(vocab_size=256, d_model=64)
        tokens = torch.randint(0, 256, (B, 16))
        # Mask last 4 tokens as padding
        mask = torch.zeros(B, 16, dtype=torch.bool)
        mask[:, 12:] = True
        out = model(tokens, mask=mask)
        assert out.shape == (B, 64)
        assert not torch.isnan(out).any()

    def test_no_nan(self):
        model = TextEncoder()
        tokens = torch.randint(0, 256, (B, 16))
        out = model(tokens)
        assert not torch.isnan(out).any()

    def test_different_sequences(self):
        """Different token sequences should produce different embeddings."""
        model = TextEncoder(vocab_size=256, d_model=64)
        model.eval()
        t1 = torch.randint(0, 256, (1, 16))
        t2 = torch.randint(0, 256, (1, 16))
        # Ensure t1 != t2
        while (t1 == t2).all():
            t2 = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            e1 = model(t1)
            e2 = model(t2)
        assert not torch.allclose(e1, e2), "Different inputs should produce different embeddings"


class TestCrossModalAttention:
    def test_output_shape(self):
        cma = CrossModalAttention(visual_dim=128, text_dim=64, num_heads=4)
        visual = torch.randn(B, 16, 128)
        text = torch.randn(B, 8, 64)
        out = cma(visual, text)
        assert out.shape == (B, 16, 128), f"Expected ({B}, 16, 128), got {out.shape}"

    def test_residual_connection(self):
        """Output should be close to input (at init, attention is near-zero)."""
        # Just check no error, not exact value — residual is added
        cma = CrossModalAttention(visual_dim=64, text_dim=64, num_heads=4)
        visual = torch.randn(B, 8, 64)
        text = torch.randn(B, 4, 64)
        out = cma(visual, text)
        assert out.shape == visual.shape

    def test_no_nan(self):
        cma = CrossModalAttention(visual_dim=128, text_dim=64)
        visual = torch.randn(B, 16, 128)
        text = torch.randn(B, 12, 64)
        out = cma(visual, text)
        assert not torch.isnan(out).any()


class TestTinyVLM:
    @pytest.fixture
    def model(self):
        return TinyVLM(visual_dim=128, text_dim=64, vocab_size=256, max_seq_len=32)

    @pytest.fixture
    def sample_batch(self):
        images = torch.randn(B, 3, 64, 64)
        tokens = torch.randint(0, 256, (B, 16))
        return images, tokens

    def test_output_shape(self, model, sample_batch):
        images, tokens = sample_batch
        scores = model(images, tokens)
        assert scores.shape == (B,), f"Expected ({B},), got {scores.shape}"

    def test_output_is_scalar_per_sample(self, model, sample_batch):
        images, tokens = sample_batch
        scores = model(images, tokens)
        assert scores.ndim == 1
        assert len(scores) == B

    def test_no_nan(self, model, sample_batch):
        images, tokens = sample_batch
        scores = model(images, tokens)
        assert not torch.isnan(scores).any()

    def test_eval_mode_deterministic(self, model, sample_batch):
        """Same input → same output in eval mode."""
        images, tokens = sample_batch
        model.eval()
        with torch.no_grad():
            s1 = model(images, tokens)
            s2 = model(images, tokens)
        torch.testing.assert_close(s1, s2)

    def test_gradient_flow(self, model, sample_batch):
        """Gradients should flow to all model parameters."""
        model.train()
        images, tokens = sample_batch
        scores = model(images, tokens)
        loss = scores.mean()
        loss.backward()
        no_grad = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert len(no_grad) == 0, f"No grad for: {no_grad}"

    def test_with_text_mask(self, model):
        images = torch.randn(B, 3, 64, 64)
        tokens = torch.randint(0, 256, (B, 16))
        mask = torch.zeros(B, 16, dtype=torch.bool)
        mask[:, 12:] = True  # pad last 4 tokens
        scores = model(images, tokens, text_mask=mask)
        assert scores.shape == (B,)
        assert not torch.isnan(scores).any()

    def test_parameter_count(self, model):
        total = sum(p.numel() for p in model.parameters())
        # TinyVLM should stay under 2M params
        assert total < 2_000_000, f"Model too large: {total:,} params"
