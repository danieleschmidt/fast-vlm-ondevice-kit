# ADR-002: Swift API Design for FastVLMKit

## Status
**Accepted** - January 2025

## Context

The Swift package must provide an intuitive, performant, and Swift-idiomatic API for iOS/macOS developers. The API design significantly impacts developer adoption, application performance, and long-term maintainability.

### Requirements
- **Simplicity**: <10 lines of code for basic integration
- **Performance**: Async/await support with streaming capabilities
- **Type Safety**: Strong typing with compile-time error prevention
- **iOS Patterns**: Follow Apple's framework design guidelines
- **Extensibility**: Support for advanced use cases and customization

### Options Considered

1. **Completion Handler Pattern**
   ```swift
   vlm.answer(image: image, question: question) { result, error in }
   ```
   - Pros: Traditional iOS pattern, familiar to developers
   - Cons: Callback hell, error-prone, outdated approach

2. **Combine Framework**
   ```swift
   vlm.answer(image: image, question: question)
       .sink(receiveValue: { answer in })
   ```
   - Pros: Reactive programming, powerful composition
   - Cons: Learning curve, overkill for simple use cases

3. **Async/Await with Actors (Selected)**
   ```swift
   let answer = try await vlm.answer(image: image, question: question)
   ```
   - Pros: Modern Swift, excellent ergonomics, safe concurrency
   - Cons: iOS 15+ requirement, requires understanding of async/await

4. **SwiftUI-First Design**
   ```swift
   @StateObject var vlm = FastVLM()
   Text(vlm.answer(for: image, question: question))
   ```
   - Pros: Perfect SwiftUI integration
   - Cons: Limited to SwiftUI, less flexible for UIKit

## Decision

Implement **Async/Await with Actor-based concurrency** as the primary API, with additional convenience methods for specific use cases.

### Core API Design

```swift
@MainActor
public class FastVLM: ObservableObject {
    // Primary inference method
    public func answer(image: UIImage, question: String) async throws -> String
    
    // Batch processing
    public func batchAnswer(images: [UIImage], questions: [String]) async throws -> [String]
    
    // Streaming inference
    public func streamAnswer(image: UIImage, question: String) -> AsyncThrowingStream<String, Error>
    
    // Configuration and lifecycle
    public init(modelPath: String, configuration: FastVLMConfiguration = .default) async throws
    public func warmup() async throws
    public func clearCache()
}

// Configuration object
public struct FastVLMConfiguration {
    public let computeUnits: MLComputeUnits
    public let maxSequenceLength: Int
    public let temperature: Float
    public let topK: Int
    
    public static let `default`: FastVLMConfiguration
    public static let highPerformance: FastVLMConfiguration
    public static let lowMemory: FastVLMConfiguration
}

// Result types with metadata
public struct VLMResult {
    public let answer: String
    public let confidence: Float
    public let processingTime: TimeInterval
    public let memoryUsage: Int64
}
```

### Rationale

**Async/Await Choice**
- Modern Swift concurrency provides excellent ergonomics
- Naturally handles long-running ML inference operations
- Eliminates callback complexity and memory management issues
- Integrates seamlessly with SwiftUI and UIKit

**Actor-Based Safety**
- Prevents data races in ML model access
- Ensures thread-safe operations without manual synchronization
- Leverages Swift's compile-time concurrency checking
- Simplifies memory management for Core ML models

**ObservableObject Integration**
- Natural SwiftUI integration with @StateObject and @ObservedObject
- Automatic UI updates when operations complete
- Supports reactive programming patterns when needed
- Maintains compatibility with Combine when required

### API Design Principles

1. **Progressive Disclosure**
   ```swift
   // Simple: Basic inference
   let answer = try await vlm.answer(image: image, question: question)
   
   // Advanced: With configuration and metadata
   let result = try await vlm.answerWithMetadata(
       image: image,
       question: question,
       configuration: .highPerformance
   )
   ```

2. **Type Safety**
   ```swift
   // Compile-time validation of image formats
   extension UIImage {
       var isValidForVLM: Bool { /* validation */ }
   }
   
   // Strongly typed configuration
   public enum ComputeStrategy {
       case automatic
       case cpuOnly
       case neuralEngine
       case balanced(priority: Priority)
   }
   ```

3. **Error Handling**
   ```swift
   public enum FastVLMError: LocalizedError {
       case modelNotLoaded
       case invalidImage(reason: String)
       case inferenceTimeout
       case insufficientMemory
       case coreMLError(MLError)
       
       public var errorDescription: String? { /* localized descriptions */ }
   }
   ```

4. **Resource Management**
   ```swift
   // Automatic resource cleanup
   deinit {
       clearCache()
       releaseModel()
   }
   
   // Manual control when needed
   public func preloadModel() async throws
   public func unloadModel() async
   ```

## Consequences

### Positive
- **Developer Experience**: Intuitive API reducing integration time
- **Performance**: Efficient async operations without blocking main thread
- **Safety**: Actor-based concurrency prevents common threading bugs
- **Compatibility**: Works with both SwiftUI and UIKit applications
- **Maintainability**: Clean separation of concerns and testable design

### Negative
- **iOS Version**: Requires iOS 15+ for async/await features
- **Learning Curve**: Developers unfamiliar with modern Swift concurrency
- **Complexity**: Internal actor management adds implementation complexity
- **Testing**: Async testing requires additional test infrastructure
- **Memory**: ObservableObject may retain objects longer than necessary

### Risks & Mitigations

**Risk**: Developers struggle with async/await patterns
- **Mitigation**: Comprehensive documentation with examples, migration guides

**Risk**: Performance issues with frequent async operations
- **Mitigation**: Intelligent batching, caching, and performance monitoring

**Risk**: Memory pressure from ObservableObject retention
- **Mitigation**: Weak references where appropriate, manual cleanup methods

### Usage Examples

**Basic Integration**
```swift
import FastVLMKit

class ViewController: UIViewController {
    private let vlm = FastVLM(modelPath: "FastVLM.mlpackage")
    
    @IBAction func analyzeImage() {
        Task {
            do {
                let answer = try await vlm.answer(
                    image: imageView.image!,
                    question: "What objects are visible?"
                )
                await MainActor.run {
                    answerLabel.text = answer
                }
            } catch {
                handleError(error)
            }
        }
    }
}
```

**SwiftUI Integration**
```swift
struct ContentView: View {
    @StateObject private var vlm = FastVLM()
    @State private var selectedImage: UIImage?
    @State private var question = ""
    @State private var answer = ""
    
    var body: some View {
        VStack {
            ImagePicker(image: $selectedImage)
            TextField("Ask a question", text: $question)
            
            Button("Analyze") {
                Task {
                    answer = try await vlm.answer(
                        image: selectedImage!,
                        question: question
                    )
                }
            }
            
            Text(answer)
        }
    }
}
```

**Advanced Usage**
```swift
// Streaming inference for real-time feedback
for try await partialAnswer in vlm.streamAnswer(image: image, question: question) {
    await MainActor.run {
        updateUI(partialAnswer)
    }
}

// Batch processing for efficiency
let results = try await vlm.batchAnswer(
    images: photoLibrary.images,
    questions: photoLibrary.questions
)
```

## Implementation Guidelines

### Performance Considerations
- Implement intelligent caching for recently processed images
- Use background queues for image preprocessing
- Batch operations when possible to amortize Core ML overhead
- Monitor memory usage and implement automatic cleanup

### Testing Strategy
- Unit tests for all public APIs with mock Core ML models
- Integration tests with real models on device
- Performance tests measuring latency and memory usage
- Concurrency tests ensuring thread safety

### Documentation Requirements
- Complete API documentation with examples
- Migration guide from completion handler patterns
- Performance optimization guide
- Troubleshooting section for common issues

## Future Enhancements

- **Combine Integration**: Optional Combine publishers for reactive programming
- **SwiftUI Modifiers**: Direct view modifiers for common VLM operations
- **Background Processing**: App background execution for long-running tasks
- **Custom Models**: Generic API supporting custom Core ML models