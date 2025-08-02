# ADR-0003: Swift API Design Principles

## Status
Accepted

## Context

FastVLM Kit requires a Swift API that provides easy integration for iOS developers while maintaining performance and following iOS development best practices. The API must balance simplicity with flexibility for different use cases.

Key considerations:
- Async/await support for modern Swift development
- Memory efficiency for mobile constraints
- Error handling for model loading and inference
- Type safety and developer experience
- Integration with Core ML and Vision frameworks

## Decision

We will design the Swift API with the following principles:

1. **Async/Await First**
   ```swift
   let answer = try await vlm.answer(image: image, question: question)
   ```
   - Modern Swift concurrency patterns
   - Non-blocking UI thread
   - Natural error propagation

2. **Protocol-Based Design**
   ```swift
   protocol VisionLanguageModel {
       func answer(image: UIImage, question: String) async throws -> String
   }
   ```
   - Testability and mocking support
   - Multiple implementation support
   - Clear API contracts

3. **Structured Configuration**
   ```swift
   struct FastVLMConfig {
       let modelPath: String
       let computeUnits: MLComputeUnits
       let maxSequenceLength: Int
   }
   ```
   - Type-safe configuration
   - Clear parameter documentation
   - Validation at initialization

4. **Resource Management**
   - Automatic model lifecycle management
   - Memory pool optimization
   - Background queue processing

## Consequences

### Positive
- Modern Swift API following platform conventions
- Excellent developer experience and type safety
- Optimal performance with async/await patterns
- Easy testing and mocking capabilities
- Clear error handling and debugging

### Negative
- Requires iOS 15+ for async/await support
- Slightly more complex than callback-based APIs
- Protocol overhead for simple use cases
- Additional abstraction layer complexity

## Alternatives Considered

1. **Callback-Based API**: Broader iOS version support but less modern
2. **Combine Framework**: Reactive patterns but learning curve for developers
3. **Synchronous API**: Simpler but blocks UI thread
4. **Objective-C Compatibility**: Broader compatibility but less type safety

## Related Documents
- [Swift Implementation](../../ios/Sources/FastVLMKit/FastVLM.swift)
- [API Documentation](../api/api-documentation.md)
- [Usage Examples](../../README.md#swift-integration)