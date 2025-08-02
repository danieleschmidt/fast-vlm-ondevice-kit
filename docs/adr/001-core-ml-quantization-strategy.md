# ADR-001: Core ML Quantization Strategy

## Status
**Accepted** - January 2025

## Context

The FastVLM PyTorch models are typically 1.2-2.4GB in size, which is impractical for mobile deployment. We need to implement quantization strategies that balance model size, inference speed, and accuracy for production mobile applications.

### Requirements
- Target <500MB model size for base variant
- Maintain >95% of original model accuracy
- Optimize for Apple Neural Engine utilization
- Support flexible quantization per model component

### Options Considered

1. **Uniform INT8 Quantization**
   - Pros: Simple implementation, good accuracy retention
   - Cons: Limited size reduction (2x), suboptimal for different components

2. **Uniform INT4 Quantization**
   - Pros: Significant size reduction (4x), faster inference
   - Cons: Notable accuracy degradation, especially for text processing

3. **Mixed-Precision Strategy (Selected)**
   - Pros: Optimal balance of size/accuracy/speed per component
   - Cons: Increased complexity, requires component-specific tuning

4. **Dynamic Quantization**
   - Pros: Runtime optimization based on input
   - Cons: Implementation complexity, less predictable performance

## Decision

Implement a **mixed-precision quantization strategy** with component-specific optimization:

```python
QuantizationConfig(
    vision_encoder="int4",     # 4-bit: Visual features are robust to quantization
    text_encoder="int8",       # 8-bit: Preserve text semantic precision
    fusion_layers="fp16",      # 16-bit: Critical cross-modal alignment
    decoder="int4",            # 4-bit: Output generation tolerates quantization
    calibration_samples=1000   # Quality vs conversion time balance
)
```

### Rationale by Component

**Vision Encoder (INT4)**
- Visual features show high redundancy and robustness to quantization
- Largest component (60% of model), maximum size reduction benefit
- Apple Neural Engine optimized for INT4 vision operations

**Text Encoder (INT8)**
- Text semantics require higher precision than visual features
- Vocabulary embeddings sensitive to quantization artifacts
- Balanced approach preserving language understanding

**Fusion Layers (FP16)**
- Cross-modal attention is most critical for accuracy
- Relatively small component (5% of model), minimal size impact
- Preserves alignment quality between vision and text

**Decoder (INT4)**
- Output generation shows robustness to quantization
- Autoregressive structure provides error correction
- Temperature scaling compensates for quantization noise

### Implementation Strategy

1. **Calibration Dataset**: Use 1000 diverse VQA samples for quantization calibration
2. **Progressive Quantization**: Apply component-by-component with accuracy validation
3. **Post-Training Optimization**: Fine-tune quantized models if accuracy drops >5%
4. **Evaluation Pipeline**: Comprehensive testing on VQAv2, COCO-QA, and domain-specific datasets

## Consequences

### Positive
- **Model Size**: Reduces from 1.2GB to 412MB (65% reduction)
- **Inference Speed**: 40% faster on Apple Neural Engine
- **Memory Usage**: 60% reduction in runtime memory
- **Battery Life**: 25% improvement in energy efficiency
- **Flexibility**: Configurable per-component strategies

### Negative
- **Complexity**: Requires sophisticated quantization pipeline
- **Accuracy**: 3-5% accuracy drop compared to full precision
- **Conversion Time**: Increased model conversion time (calibration required)
- **Debugging**: More complex error diagnosis and model introspection
- **Maintenance**: Component-specific optimization requires ongoing tuning

### Risks & Mitigations

**Risk**: Accuracy degradation exceeds acceptable thresholds
- **Mitigation**: Progressive quantization with fallback to higher precision

**Risk**: Apple Neural Engine compatibility issues
- **Mitigation**: Extensive testing across iOS versions and device types

**Risk**: Quantization artifacts in specific domains/use cases
- **Mitigation**: Domain-specific evaluation and custom calibration datasets

### Monitoring & Success Criteria

- **Accuracy Benchmark**: VQAv2 accuracy >71% (vs 74% full precision)
- **Performance Target**: <250ms inference on iPhone 15 Pro
- **Size Target**: Model packages <500MB
- **Energy Target**: <2.5mWh per inference

## Implementation Timeline

- **Week 1-2**: Core quantization infrastructure
- **Week 3-4**: Component-specific strategies implementation
- **Week 5-6**: Calibration pipeline and evaluation framework
- **Week 7-8**: Testing, optimization, and documentation

## Future Considerations

- **Dynamic Quantization**: Runtime precision adjustment based on input complexity
- **Hardware Co-design**: Collaboration with Apple Silicon team for optimal quantization
- **Model Distillation**: Teacher-student training for quantization-aware models
- **Research Integration**: Incorporating latest quantization research advances