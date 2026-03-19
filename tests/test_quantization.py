"""Tests for INT8 quantization simulation."""

import pytest
import torch
import torch.nn as nn

from fast_vlm_ondevice.models import TinyVLM
from fast_vlm_ondevice.quantization import QuantizationSimulator, QuantizationReport


class TestQuantizationSimulator:
    @pytest.fixture
    def tiny_model(self):
        return TinyVLM(visual_dim=64, text_dim=32, vocab_size=64, max_seq_len=16)

    @pytest.fixture
    def sim(self):
        return QuantizationSimulator(bits=8)

    def test_returns_model_and_report(self, sim, tiny_model):
        q_model, report = sim.quantize(tiny_model)
        assert isinstance(q_model, nn.Module)
        assert isinstance(report, QuantizationReport)

    def test_does_not_modify_original(self, sim, tiny_model):
        """Default quantize() should copy, not modify."""
        original_weights = {
            k: v.clone() for k, v in tiny_model.state_dict().items()
        }
        sim.quantize(tiny_model, inplace=False)
        for k, v in tiny_model.state_dict().items():
            torch.testing.assert_close(v, original_weights[k])

    def test_inplace_modifies_model(self, sim, tiny_model):
        """inplace=True should modify the model."""
        original_weights = {
            k: v.clone() for k, v in tiny_model.state_dict().items()
        }
        sim.quantize(tiny_model, inplace=True)
        any_changed = any(
            not torch.allclose(v, original_weights[k])
            for k, v in tiny_model.state_dict().items()
        )
        assert any_changed, "inplace=True should modify weights"

    def test_quantized_params_count(self, sim, tiny_model):
        _, report = sim.quantize(tiny_model)
        assert report.quantized_params > 0
        assert report.total_params > 0
        assert report.quantized_params <= report.total_params

    def test_weight_error_nonnegative(self, sim, tiny_model):
        _, report = sim.quantize(tiny_model)
        assert report.mean_weight_error >= 0
        assert report.max_weight_error >= 0

    def test_snr_positive(self, sim, tiny_model):
        _, report = sim.quantize(tiny_model)
        # Non-zero weights should have positive SNR
        assert report.snr_db > 0

    def test_layer_errors_populated(self, sim, tiny_model):
        _, report = sim.quantize(tiny_model)
        assert len(report.layer_errors) > 0

    def test_report_str(self, sim, tiny_model):
        _, report = sim.quantize(tiny_model)
        s = str(report)
        assert "QuantizationReport" in s
        assert "Parameters quantized" in s

    def test_compare_outputs_returns_dict(self, sim, tiny_model):
        q_model, _ = sim.quantize(tiny_model)
        images = torch.randn(2, 3, 64, 64)
        tokens = torch.randint(0, 64, (2, 16))
        result = sim.compare_outputs(tiny_model, q_model, (images, tokens))
        assert "mean_abs_diff" in result
        assert "max_abs_diff" in result
        assert "cosine_similarity" in result

    def test_compare_outputs_values_are_floats(self, sim, tiny_model):
        q_model, _ = sim.quantize(tiny_model)
        images = torch.randn(2, 3, 64, 64)
        tokens = torch.randint(0, 64, (2, 16))
        result = sim.compare_outputs(tiny_model, q_model, (images, tokens))
        for k, v in result.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

    def test_simple_linear_quantization(self, sim):
        """Verify quantization math on a known simple module."""
        linear = nn.Linear(4, 2, bias=False)
        nn.init.constant_(linear.weight, 0.5)
        q_linear, report = sim.quantize(linear)
        # 0.5 should round cleanly to a nearby representable value
        assert report.mean_weight_error < 0.01

    def test_int4_quantization(self):
        """INT4 should have higher error than INT8."""
        tiny = TinyVLM(visual_dim=64, text_dim=32, vocab_size=64, max_seq_len=16)
        sim8 = QuantizationSimulator(bits=8)
        sim4 = QuantizationSimulator(bits=4)
        _, report8 = sim8.quantize(tiny)
        _, report4 = sim4.quantize(tiny)
        assert report4.mean_weight_error >= report8.mean_weight_error
