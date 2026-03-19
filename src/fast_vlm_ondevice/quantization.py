"""
INT8 Quantization Simulator for edge deployment.

Simulates the effect of 8-bit integer quantization on model weights and
activations without requiring actual hardware-level quantization. Useful for:
  - Estimating accuracy degradation before real quantization
  - Understanding which layers are most quantization-sensitive
  - Comparing FP32 vs INT8 performance characteristics
"""

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class QuantizationReport:
    """Results from quantizing a model."""
    layer_errors: dict[str, float] = field(default_factory=dict)
    total_params: int = 0
    quantized_params: int = 0
    mean_weight_error: float = 0.0
    max_weight_error: float = 0.0
    snr_db: float = 0.0  # signal-to-noise ratio of weight approximation

    def __str__(self) -> str:
        lines = [
            "QuantizationReport (FP32 → INT8)",
            f"  Parameters quantized: {self.quantized_params:,} / {self.total_params:,}",
            f"  Mean weight error:    {self.mean_weight_error:.6f}",
            f"  Max weight error:     {self.max_weight_error:.6f}",
            f"  Weight SNR:           {self.snr_db:.2f} dB",
        ]
        if self.layer_errors:
            lines.append("  Per-layer mean errors:")
            for name, err in sorted(self.layer_errors.items()):
                lines.append(f"    {name:40s} {err:.6f}")
        return "\n".join(lines)


class QuantizationSimulator:
    """
    Simulates INT8 quantization on a PyTorch model.

    Strategy: per-tensor symmetric quantization
        - Compute scale = max(|w|) / 127
        - Quantize:  w_q = round(clip(w / scale, -127, 127))
        - Dequantize: w_approx = w_q * scale
        - Error: |w - w_approx|

    Usage::

        model = TinyVLM()
        sim = QuantizationSimulator()
        q_model, report = sim.quantize(model)
        print(report)
    """

    def __init__(self, bits: int = 8, skip_bn: bool = True):
        """
        Args:
            bits: Quantization bit width (default 8 for INT8).
            skip_bn: Skip BatchNorm layers (they're usually folded at inference).
        """
        assert bits in (4, 8, 16), f"Unsupported bit width: {bits}"
        self.bits = bits
        self.skip_bn = skip_bn
        self._max_val = float(2 ** (bits - 1) - 1)  # 127 for INT8

    def _quantize_tensor(self, w: torch.Tensor) -> torch.Tensor:
        """Apply symmetric per-tensor quantization and dequantize."""
        abs_max = w.abs().max().clamp(min=1e-9)
        scale = abs_max / self._max_val
        w_q = (w / scale).round().clamp(-self._max_val, self._max_val)
        return w_q * scale  # dequantized (same dtype as original)

    def quantize(
        self, model: nn.Module, inplace: bool = False
    ) -> tuple[nn.Module, QuantizationReport]:
        """
        Simulate INT8 quantization of all eligible weight tensors.

        Args:
            model: Source model (not modified unless inplace=True).
            inplace: Modify model in place (default False — returns a copy).

        Returns:
            (quantized_model, QuantizationReport)
        """
        q_model = model if inplace else copy.deepcopy(model)
        report = QuantizationReport()

        total_error_sum = 0.0
        total_error_max = 0.0
        total_signal_power = 0.0
        total_noise_power = 0.0

        for name, module in q_model.named_modules():
            # Optionally skip BatchNorm
            if self.skip_bn and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                continue

            layer_errors = []
            for param_name, param in module.named_parameters(recurse=False):
                if param.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    continue

                report.total_params += param.numel()
                report.quantized_params += param.numel()

                with torch.no_grad():
                    original = param.data.clone()
                    q_data = self._quantize_tensor(original)
                    param.data.copy_(q_data)

                error = (original - q_data).abs()
                layer_errors.append(error.mean().item())
                total_error_sum += error.mean().item()
                total_error_max = max(total_error_max, error.max().item())

                # SNR components
                total_signal_power += original.pow(2).mean().item()
                total_noise_power += error.pow(2).mean().item()

            if layer_errors and name:
                report.layer_errors[name] = float(torch.tensor(layer_errors).mean())

        # Aggregate stats
        n = len(report.layer_errors) or 1
        report.mean_weight_error = total_error_sum / n
        report.max_weight_error = total_error_max

        if total_noise_power > 0:
            import math
            report.snr_db = 10 * math.log10(
                total_signal_power / total_noise_power + 1e-12
            )

        return q_model, report

    def compare_outputs(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        sample_inputs: tuple,
    ) -> dict[str, float]:
        """
        Compare outputs of original vs quantized model on the same inputs.

        Args:
            original_model: FP32 model.
            quantized_model: INT8-simulated model.
            sample_inputs: Tuple of tensors passed directly to forward().

        Returns:
            dict with keys: mean_abs_diff, max_abs_diff, cosine_similarity
        """
        original_model.eval()
        quantized_model.eval()

        with torch.no_grad():
            out_fp32 = original_model(*sample_inputs)
            out_int8 = quantized_model(*sample_inputs)

        diff = (out_fp32 - out_int8).abs()
        cos_sim = F.cosine_similarity(
            out_fp32.unsqueeze(0), out_int8.unsqueeze(0)
        ).item()

        return {
            "mean_abs_diff": diff.mean().item(),
            "max_abs_diff": diff.max().item(),
            "cosine_similarity": cos_sim,
        }


# Avoid circular import — import F here for compare_outputs
import torch.nn.functional as F  # noqa: E402
