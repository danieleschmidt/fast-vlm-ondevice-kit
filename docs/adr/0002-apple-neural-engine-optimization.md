# ADR-0002: Apple Neural Engine Optimization

## Status
Accepted

## Context

Apple devices include a dedicated Neural Engine (ANE) that provides high-performance, energy-efficient ML computation. To achieve optimal performance for FastVLM on iOS devices, we need to ensure our models utilize the ANE effectively.

The ANE has specific constraints and preferences:
- Optimized for specific operator types
- Limited memory bandwidth between CPU and ANE
- Works best with quantized models
- Requires specific tensor layouts and shapes

## Decision

We will optimize FastVLM models specifically for Apple Neural Engine deployment:

1. **Core ML Optimization**
   - Use `coremltools` with ANE-specific optimization passes
   - Set `compute_units="ALL"` to enable ANE utilization
   - Implement ANE-preferred operator patterns

2. **Model Architecture Adaptations**
   - Use ANE-optimized attention mechanisms
   - Implement efficient matrix multiplication patterns
   - Minimize data movement between compute units

3. **Memory Layout Optimization**
   - Optimize tensor layouts for ANE memory access patterns
   - Use Metal Performance Shaders for data preprocessing
   - Implement efficient batching strategies

4. **Performance Monitoring**
   - Implement ANE utilization monitoring
   - Track thermal management and power consumption
   - Measure end-to-end latency including data transfers

## Consequences

### Positive
- 3-4x performance improvement over CPU-only inference
- 50% reduction in energy consumption
- Thermal efficiency allows sustained performance
- Enables real-time inference on mobile devices
- Better user experience with faster response times

### Negative
- Increased complexity in model conversion
- ANE-specific optimizations limit portability
- Debugging and profiling more challenging
- Dependency on Apple-specific toolchain
- Model architecture constraints for ANE compatibility

## Alternatives Considered

1. **CPU-Only Deployment**: Simpler but 4x slower performance
2. **GPU Acceleration**: Good performance but higher energy consumption
3. **Hybrid CPU-GPU**: Complex scheduling, inconsistent performance
4. **Custom Metal Kernels**: High performance but significant development effort

## Related Documents
- [Core ML Integration](../../ios/Sources/FastVLMKit/FastVLM.swift)
- [Performance Benchmarks](../operations/performance-benchmarking.md)
- [Model Quantization Strategy](./0001-model-quantization-strategy.md)