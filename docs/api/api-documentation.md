# API Documentation

This document provides comprehensive API documentation for Fast VLM On-Device Kit, covering both Python and Swift APIs.

## Python API Reference

### Core Classes

#### FastVLMConverter

```python
class FastVLMConverter:
    """Main class for converting PyTorch FastVLM models to Core ML format."""
    
    def __init__(self, device: str = "auto", cache_dir: Optional[str] = None):
        """
        Initialize the FastVLM converter.
        
        Args:
            device: Target device ('cpu', 'gpu', 'auto')
            cache_dir: Directory for caching intermediate results
        """
        
    def load_pytorch_model(self, checkpoint_path: str) -> torch.nn.Module:
        """
        Load a PyTorch FastVLM model from checkpoint.
        
        Args:
            checkpoint_path: Path to the PyTorch model checkpoint (.pth file)
            
        Returns:
            Loaded PyTorch model
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ModelLoadError: If model format is invalid
            
        Example:
            >>> converter = FastVLMConverter()
            >>> model = converter.load_pytorch_model("models/fast-vlm-base.pth")
        """
        
    def convert_to_coreml(
        self,
        model: torch.nn.Module,
        quantization: str = "int4",
        compute_units: str = "ALL",
        image_size: Tuple[int, int] = (336, 336),
        max_seq_length: int = 77
    ) -> coremltools.models.MLModel:
        """
        Convert PyTorch model to Core ML format.
        
        Args:
            model: PyTorch model to convert
            quantization: Quantization level ('int4', 'int8', 'fp16', 'fp32')
            compute_units: Target compute units ('CPU_ONLY', 'CPU_AND_GPU', 'ALL')
            image_size: Input image dimensions (height, width)
            max_seq_length: Maximum text sequence length
            
        Returns:
            Converted Core ML model
            
        Raises:
            ConversionError: If conversion fails
            UnsupportedModelError: If model architecture not supported
            
        Example:
            >>> coreml_model = converter.convert_to_coreml(
            ...     model,
            ...     quantization="int4",
            ...     compute_units="ALL",
            ...     image_size=(336, 336)
            ... )
        """
        
    def get_model_size_mb(self) -> float:
        """
        Get the size of the last converted model in MB.
        
        Returns:
            Model size in megabytes
        """
        
    def benchmark_model(
        self,
        model: coremltools.models.MLModel,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            model: Core ML model to benchmark
            num_iterations: Number of inference iterations
            
        Returns:
            Dictionary containing performance metrics:
            - avg_latency_ms: Average inference latency
            - min_latency_ms: Minimum latency
            - max_latency_ms: Maximum latency
            - throughput_qps: Queries per second
            
        Example:
            >>> metrics = converter.benchmark_model(coreml_model, 50)
            >>> print(f"Average latency: {metrics['avg_latency_ms']:.1f}ms")
        """
```

#### QuantizationConfig

```python
class QuantizationConfig:
    """Configuration for model quantization strategies."""
    
    def __init__(
        self,
        vision_encoder: str = "int4",
        text_encoder: str = "int8", 
        fusion_layers: str = "fp16",
        decoder: str = "int4",
        calibration_samples: int = 1000
    ):
        """
        Initialize quantization configuration.
        
        Args:
            vision_encoder: Quantization for vision encoder ('int4', 'int8', 'fp16')
            text_encoder: Quantization for text encoder
            fusion_layers: Quantization for fusion layers
            decoder: Quantization for decoder
            calibration_samples: Number of samples for calibration
        """
        
    @classmethod
    def preset(cls, preset_name: str) -> 'QuantizationConfig':
        """
        Load a preset quantization configuration.
        
        Args:
            preset_name: Name of preset ('balanced', 'aggressive', 'conservative')
            
        Returns:
            Pre-configured QuantizationConfig instance
            
        Available presets:
        - 'balanced': Good trade-off between size and accuracy
        - 'aggressive': Maximum compression, some accuracy loss
        - 'conservative': Minimal accuracy loss, larger size
            
        Example:
            >>> config = QuantizationConfig.preset('balanced')
        """
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantizationConfig':
        """Create configuration from dictionary."""
```

### Utility Functions

#### Model Management

```python
def download_model(
    model_name: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False
) -> str:
    """
    Download a pre-trained FastVLM model.
    
    Args:
        model_name: Name of model ('fast-vlm-tiny', 'fast-vlm-base', 'fast-vlm-large')
        cache_dir: Directory to cache downloaded models
        force_download: Force re-download even if cached
        
    Returns:
        Path to downloaded model file
        
    Example:
        >>> model_path = download_model('fast-vlm-base')
        >>> model = converter.load_pytorch_model(model_path)
    """

def list_available_models() -> List[Dict[str, Any]]:
    """
    List all available pre-trained models.
    
    Returns:
        List of model information dictionaries containing:
        - name: Model name
        - size: Model size category
        - parameters: Number of parameters  
        - accuracy: Reported accuracy on VQAv2
        - file_size_mb: Download size in MB
        
    Example:
        >>> models = list_available_models()
        >>> for model in models:
        ...     print(f"{model['name']}: {model['accuracy']:.1%} accuracy")
    """
```

#### Performance Monitoring

```python
def profile_conversion(
    converter: FastVLMConverter,
    model_path: str,
    output_path: str,
    config: QuantizationConfig
) -> Dict[str, Any]:
    """
    Profile model conversion performance.
    
    Args:
        converter: FastVLMConverter instance
        model_path: Path to input PyTorch model
        output_path: Path for output Core ML model
        config: Quantization configuration
        
    Returns:
        Profiling results including:
        - conversion_time_s: Total conversion time
        - peak_memory_mb: Peak memory usage
        - model_size_reduction: Size reduction ratio
        - accuracy_retention: Estimated accuracy retention
        
    Example:
        >>> profile = profile_conversion(converter, "model.pth", "model.mlpackage", config)
        >>> print(f"Conversion took {profile['conversion_time_s']:.1f}s")
    """
```

## Swift API Reference

### Core Classes

#### FastVLM

```swift
/// Main class for on-device VLM inference
public class FastVLM {
    
    /// Initialize FastVLM with model path
    /// - Parameter modelPath: Path to Core ML model (.mlpackage)
    /// - Throws: FastVLMError if model loading fails
    public init(modelPath: String) throws
    
    /// Answer a question about an image
    /// - Parameters:
    ///   - image: Input image
    ///   - question: Question to answer
    /// - Returns: Generated answer text
    /// - Throws: FastVLMError if inference fails
    public func answer(image: UIImage, question: String) async throws -> String
    
    /// Answer multiple questions about an image (batch processing)
    /// - Parameters:
    ///   - image: Input image
    ///   - questions: Array of questions
    /// - Returns: Array of answers corresponding to questions
    /// - Throws: FastVLMError if inference fails
    public func batchAnswer(image: UIImage, questions: [String]) async throws -> [String]
    
    /// Stream answer generation (word by word)
    /// - Parameters:
    ///   - image: Input image
    ///   - question: Question to answer
    /// - Returns: AsyncSequence of answer tokens
    /// - Throws: FastVLMError if inference fails
    public func streamAnswer(image: UIImage, question: String) -> AsyncThrowingStream<String, Error>
    
    /// Get model information
    /// - Returns: Dictionary containing model metadata
    public func getModelInfo() -> [String: Any]
}
```

#### FastVLMConfig

```swift
/// Configuration for FastVLM inference
public struct FastVLMConfig {
    /// Maximum sequence length for generated text
    public var maxSequenceLength: Int = 32
    
    /// Temperature for text generation (0.0 - 1.0)
    public var temperature: Float = 0.7
    
    /// Top-k sampling parameter
    public var topK: Int = 50
    
    /// Enable/disable Apple Neural Engine
    public var useNeuralEngine: Bool = true
    
    /// Batch size for processing multiple questions
    public var batchSize: Int = 1
    
    /// Default configuration
    public static let `default` = FastVLMConfig()
    
    /// Performance-optimized configuration
    public static let performance = FastVLMConfig(
        maxSequenceLength: 16,
        temperature: 0.1,
        topK: 10,
        useNeuralEngine: true,
        batchSize: 4
    )
    
    /// Quality-optimized configuration  
    public static let quality = FastVLMConfig(
        maxSequenceLength: 64,
        temperature: 0.8,
        topK: 100,
        useNeuralEngine: true,
        batchSize: 1
    )
}
```

#### FastVLMProfiler

```swift
/// Performance profiling utilities
public class FastVLMProfiler {
    
    /// Profile inference performance
    /// - Parameters:
    ///   - model: FastVLM instance to profile
    ///   - iterations: Number of inference iterations
    ///   - warmup: Number of warmup iterations
    /// - Returns: Performance metrics
    /// - Throws: FastVLMError if profiling fails
    public func profile(
        model: FastVLM,
        iterations: Int = 100,
        warmup: Int = 10
    ) async throws -> PerformanceMetrics
    
    /// Profile memory usage during inference
    /// - Parameters:
    ///   - model: FastVLM instance
    ///   - testImage: Image for testing
    ///   - testQuestion: Question for testing
    /// - Returns: Memory usage metrics
    public func profileMemory(
        model: FastVLM,
        testImage: UIImage,
        testQuestion: String
    ) async throws -> MemoryMetrics
}
```

### Data Types

#### PerformanceMetrics

```swift
/// Performance metrics from profiling
public struct PerformanceMetrics {
    /// Average inference latency in milliseconds
    public let avgLatencyMs: Double
    
    /// Minimum latency in milliseconds
    public let minLatencyMs: Double
    
    /// Maximum latency in milliseconds  
    public let maxLatencyMs: Double
    
    /// 95th percentile latency in milliseconds
    public let p95LatencyMs: Double
    
    /// Queries per second throughput
    public let throughputQPS: Double
    
    /// Peak memory usage in MB
    public let peakMemoryMB: Double
    
    /// Energy impact score (0.0 - 1.0)
    public let energyImpact: Double
}
```

#### MemoryMetrics

```swift
/// Memory usage metrics
public struct MemoryMetrics {
    /// Peak memory usage in bytes
    public let peakMemoryBytes: Int64
    
    /// Memory usage before inference
    public let baselineMemoryBytes: Int64
    
    /// Memory increase during inference
    public let memoryIncreaseBytes: Int64
    
    /// Memory usage after inference (for leak detection)
    public let finalMemoryBytes: Int64
}
```

#### FastVLMError

```swift
/// Errors that can occur during FastVLM operations
public enum FastVLMError: LocalizedError {
    case modelLoadError(String)
    case inferenceError(String)
    case invalidInput(String)
    case configurationError(String)
    case resourceError(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelLoadError(let message):
            return "Model loading failed: \(message)"
        case .inferenceError(let message):
            return "Inference failed: \(message)"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .configurationError(let message):
            return "Configuration error: \(message)"
        case .resourceError(let message):
            return "Resource error: \(message)"
        }
    }
}
```

## Usage Examples

### Python Examples

#### Basic Model Conversion

```python
from fast_vlm_ondevice import FastVLMConverter, QuantizationConfig

# Initialize converter
converter = FastVLMConverter()

# Load PyTorch model
model = converter.load_pytorch_model("models/fast-vlm-base.pth")

# Configure quantization
config = QuantizationConfig.preset('balanced')

# Convert to Core ML
coreml_model = converter.convert_to_coreml(
    model,
    quantization="int4",
    compute_units="ALL",
    image_size=(336, 336)
)

# Save converted model
coreml_model.save("FastVLM.mlpackage")
print(f"Model size: {converter.get_model_size_mb():.1f}MB")
```

#### Advanced Quantization

```python
from fast_vlm_ondevice import FastVLMConverter, QuantizationConfig

# Custom quantization configuration
config = QuantizationConfig(
    vision_encoder="int4",      # Aggressive compression
    text_encoder="int8",        # Moderate compression
    fusion_layers="fp16",       # Preserve precision
    decoder="int4",             # Aggressive compression
    calibration_samples=1500    # More calibration data
)

converter = FastVLMConverter()
model = converter.load_pytorch_model("models/fast-vlm-large.pth")

# Apply custom quantization
quantized_model = converter.quantize_model(model, config)

# Convert with custom settings
coreml_model = converter.convert_to_coreml(
    quantized_model,
    compute_units="CPU_AND_GPU",  # Exclude Neural Engine
    image_size=(512, 512),        # Higher resolution
    max_seq_length=128            # Longer sequences
)
```

### Swift Examples

#### Basic Inference

```swift
import FastVLMKit
import UIKit

// Initialize FastVLM
let vlm = try FastVLM(modelPath: "FastVLM.mlpackage")

// Load test image
let image = UIImage(named: "test_image.jpg")!
let question = "What objects are in this image?"

// Perform inference
let answer = try await vlm.answer(image: image, question: question)
print("Answer: \(answer)")
```

#### Batch Processing

```swift
import FastVLMKit

let vlm = try FastVLM(modelPath: "FastVLM.mlpackage")
let image = UIImage(named: "scene.jpg")!

let questions = [
    "What is the main subject?",
    "What colors are visible?", 
    "How many people are there?",
    "What time of day is it?",
    "Describe the setting."
]

// Process multiple questions efficiently
let answers = try await vlm.batchAnswer(image: image, questions: questions)

for (question, answer) in zip(questions, answers) {
    print("Q: \(question)")
    print("A: \(answer)\n")
}
```

#### Performance Profiling

```swift
import FastVLMKit

let vlm = try FastVLM(modelPath: "FastVLM.mlpackage")
let profiler = FastVLMProfiler()

// Profile performance
let metrics = try await profiler.profile(
    model: vlm,
    iterations: 100,
    warmup: 10
)

print("Performance Metrics:")
print("  Average latency: \(metrics.avgLatencyMs)ms")
print("  95th percentile: \(metrics.p95LatencyMs)ms")
print("  Throughput: \(metrics.throughputQPS) QPS")
print("  Peak memory: \(metrics.peakMemoryMB)MB")
print("  Energy impact: \(metrics.energyImpact)")
```

#### Streaming Inference

```swift
import FastVLMKit

let vlm = try FastVLM(modelPath: "FastVLM.mlpackage")
let image = UIImage(named: "photo.jpg")!
let question = "Describe this image in detail."

// Stream answer generation
for try await token in vlm.streamAnswer(image: image, question: question) {
    print(token, terminator: " ")
}
print() // New line after complete answer
```

## API Conventions

### Error Handling

- Python APIs use standard Python exceptions with descriptive messages
- Swift APIs use typed errors conforming to `LocalizedError`
- All async operations can throw errors and should be wrapped in try-catch blocks

### Threading and Concurrency

- Python APIs are synchronous by default, with async variants for long operations
- Swift APIs use modern async/await patterns for all inference operations
- Thread safety is guaranteed for all public APIs

### Memory Management

- Python APIs handle memory management automatically
- Swift APIs use ARC and provide explicit cleanup methods when needed
- Large models are loaded lazily and cached efficiently

### Backwards Compatibility

- APIs follow semantic versioning (semver)
- Deprecated methods are marked and supported for at least one major version
- Breaking changes are clearly documented in release notes

This comprehensive API documentation provides everything needed to integrate Fast VLM On-Device Kit into applications across platforms.