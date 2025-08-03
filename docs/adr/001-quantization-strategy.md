# ADR-001: Quantization Strategy for Mobile Deployment

**Status**: Accepted  
**Date**: 2025-08-03  
**Authors**: FastVLM Team  

## Context

FastVLM models require aggressive quantization to run efficiently on mobile devices while maintaining acceptable accuracy. We need to decide on quantization strategies that balance model size, inference speed, and output quality.

## Decision

We will implement a **tiered quantization strategy** with different precision levels for each model component:

### Quantization Mapping
- **Vision Encoder**: INT4 (aggressive compression)
- **Text Encoder**: INT8 (moderate compression)  
- **Fusion Layers**: FP16 (preserve critical cross-modal alignment)
- **Decoder**: INT4 (aggressive compression for generation)

### Rationale

1. **Vision Encoder (INT4)**:
   - Visual features are robust to quantization
   - Largest component, benefits most from compression
   - Pattern recognition tolerates reduced precision

2. **Text Encoder (INT8)**:
   - Text semantics require moderate precision
   - Smaller impact on overall model size
   - Vocabulary embeddings benefit from INT8

3. **Fusion Layers (FP16)**:
   - Cross-modal alignment is most sensitive to quantization
   - Critical for maintaining VLM quality
   - Small component, minimal size impact

4. **Decoder (INT4)**:
   - Generation tasks handle quantization well
   - Autoregressive nature provides self-correction
   - Significant size reduction opportunity

## Alternatives Considered

### Uniform INT4 Quantization
- **Pros**: Maximum compression, simplest implementation
- **Cons**: Significant accuracy degradation in fusion layers
- **Rejected**: Quality impact too severe for production use

### Uniform INT8 Quantization  
- **Pros**: Better quality preservation, still good compression
- **Cons**: Suboptimal size reduction, vision encoder over-preserved
- **Rejected**: Not aggressive enough for mobile constraints

### Dynamic Quantization
- **Pros**: Optimal per-layer precision automatically
- **Cons**: Complex calibration, inconsistent performance
- **Rejected**: Too complex for initial implementation

## Implementation Details

### Calibration Dataset
- 1,000 representative image-question pairs from VQAv2
- Diverse visual content and question types
- Balanced distribution across model components

### Quantization Process
```python
config = QuantizationConfig(
    vision_encoder="int4",
    text_encoder="int8", 
    fusion_layers="fp16",
    decoder="int4",
    calibration_samples=1000
)
```

### Quality Validation
- Target <2% accuracy drop on VQAv2 validation set
- Comprehensive testing across model variants
- Performance regression detection in CI

## Consequences

### Positive
- **Size Reduction**: ~75% model size reduction vs FP32
- **Speed Improvement**: 2-3x faster inference on ANE
- **Memory Efficiency**: Fits within mobile app constraints
- **Flexibility**: Per-component tuning for different use cases

### Negative  
- **Complexity**: More complex quantization pipeline
- **Validation Overhead**: Multi-tier quality testing required
- **Potential Quality Loss**: Some accuracy degradation expected
- **Calibration Dependency**: Requires representative dataset

## Monitoring

We will track the following metrics to validate this decision:

- Model size reduction percentage by component
- Inference latency improvement on target devices
- Accuracy degradation on benchmark datasets
- Memory usage during inference
- Energy consumption impact

## Future Considerations

- **Adaptive Quantization**: Per-sample or per-layer adaptation
- **Knowledge Distillation**: Training smaller models explicitly
- **Hardware-Specific**: Optimization for specific Apple Silicon variants
- **Post-Training**: Advanced post-training quantization techniques

This ADR establishes the foundation for all quantization work in the FastVLM On-Device Kit.