# ADR-0001: Model Quantization Strategy

## Status
Accepted

## Context

FastVLM models require significant memory and computational resources. To achieve <250ms inference on mobile devices, we need to implement aggressive quantization while maintaining acceptable accuracy levels.

Different model components have varying sensitivity to quantization:
- Vision encoders are generally robust to aggressive quantization
- Text encoders require moderate precision for semantic preservation
- Fusion layers are critical for cross-modal alignment
- Decoders can handle aggressive quantization for generation tasks

## Decision

We will implement a heterogeneous quantization strategy with per-component precision levels:

1. **Vision Encoder**: INT4 quantization
   - High compression ratio (4x)
   - Visual features are robust to quantization
   - Minimal accuracy impact observed

2. **Text Encoder**: INT8 quantization
   - Balanced compression (2x)
   - Preserves text semantic information
   - Good quality/performance trade-off

3. **Fusion Layers**: FP16 quantization
   - Minimal compression but high precision
   - Critical for vision-text alignment
   - Prevents quality degradation

4. **Decoder**: INT4 quantization
   - High compression ratio (4x)
   - Generation quality remains acceptable
   - Significant memory savings

## Consequences

### Positive
- 70% reduction in model size (from 1.4GB to 412MB)
- 60% reduction in memory usage during inference
- Maintains 95%+ of original model accuracy
- Fits within mobile device memory constraints
- Enables Apple Neural Engine utilization

### Negative
- Slight quality degradation compared to FP32 baseline
- Increased complexity in model conversion pipeline
- Requires calibration dataset for optimal results
- Limited flexibility for runtime precision adjustment

## Alternatives Considered

1. **Uniform INT8**: Simpler but larger model size, insufficient compression
2. **Uniform INT4**: Aggressive compression but significant accuracy loss
3. **Dynamic Quantization**: Runtime overhead, inconsistent performance
4. **Knowledge Distillation**: Requires retraining, longer development cycle

## Related Documents
- [Quantization Implementation](../../src/fast_vlm_ondevice/quantization.py)
- [Performance Benchmarks](../operations/performance-benchmarking.md)
- [Apple Neural Engine Optimization](./0002-apple-neural-engine-optimization.md)