# API Reference

Complete API documentation for Fast VLM On-Device Kit.

## Python API

### FastVLMConverter

Main class for converting PyTorch models to Core ML format.

#### Constructor

```python
converter = FastVLMConverter()
```

#### Methods

##### `load_pytorch_model(checkpoint_path: str)`

Load a FastVLM model from PyTorch checkpoint.

**Parameters:**
- `checkpoint_path` (str): Path to the .pth checkpoint file

**Returns:**
- PyTorch model object

**Example:**
```python
model = converter.load_pytorch_model("checkpoints/fast-vlm-base.pth")
```

##### `convert_to_coreml(model, quantization="int4", compute_units="ALL", image_size=(336, 336), max_seq_length=77)`

Convert PyTorch model to Core ML format with optimizations.

**Parameters:**
- `model`: PyTorch model to convert
- `quantization` (str): Quantization type - "int4", "int8", "fp16"
- `compute_units` (str): Target compute units - "ALL", "CPU_AND_GPU", "CPU_ONLY"
- `image_size` (tuple): Input image dimensions (height, width)
- `max_seq_length` (int): Maximum text sequence length

**Returns:**
- Core ML model object

**Example:**
```python
coreml_model = converter.convert_to_coreml(
    model,
    quantization="int4",
    compute_units="ALL",
    image_size=(336, 336),
    max_seq_length=77
)
```

##### `get_model_size_mb() -> float`

Get the size of the converted model in megabytes.

**Returns:**
- float: Model size in MB

### QuantizationConfig

Configuration class for advanced quantization settings.

#### Constructor

```python
from fast_vlm_ondevice.quantization import QuantizationConfig

config = QuantizationConfig(
    vision_encoder="int4",
    text_encoder="int8", 
    fusion_layers="fp16",
    decoder="int4",
    calibration_samples=1000
)
```

**Parameters:**
- `vision_encoder` (str): Quantization for vision encoder ("int4", "int8", "fp16")
- `text_encoder` (str): Quantization for text encoder
- `fusion_layers` (str): Quantization for fusion/attention layers
- `decoder` (str): Quantization for decoder layers
- `calibration_samples` (int): Number of samples for calibration

## Swift API

### FastVLM

Main class for on-device inference on iOS/macOS.

#### Constructor

```swift
import FastVLMKit

let vlm = try FastVLM(modelPath: "FastVLM.mlpackage")
```

**Parameters:**
- `modelPath`: Path to the .mlpackage file

**Throws:**
- `FastVLMError.modelLoadFailed` if model cannot be loaded
- `FastVLMError.unsupportedDevice` if device is not compatible

#### Methods

##### `answer(image: UIImage, question: String) async throws -> String`

Generate an answer for a visual question.

**Parameters:**
- `image`: Input image (UIImage)
- `question`: Question about the image

**Returns:**
- String: Generated answer

**Throws:**
- `FastVLMError.inferenceError` if inference fails
- `FastVLMError.invalidInput` if inputs are invalid

**Example:**
```swift
let image = UIImage(named: "example.jpg")!
let question = "What objects are in this image?"
let answer = try await vlm.answer(image: image, question: question)
print("Answer: \(answer)")
```

##### `batchAnswer(images: [UIImage], questions: [String]) async throws -> [String]`

Process multiple image-question pairs in batch.

**Parameters:**
- `images`: Array of input images
- `questions`: Array of questions (must match images count)

**Returns:**
- [String]: Array of generated answers

**Example:**
```swift
let images = [image1, image2, image3]
let questions = ["What is this?", "What color is it?", "How many objects?"]
let answers = try await vlm.batchAnswer(images: images, questions: questions)
```

##### `streamAnswer(image: UIImage, question: String) -> AsyncThrowingStream<String, Error>`

Stream answer generation for real-time updates.

**Parameters:**
- `image`: Input image
- `question`: Question about the image

**Returns:**
- AsyncThrowingStream yielding partial answers

**Example:**
```swift
for try await partialAnswer in vlm.streamAnswer(image: image, question: question) {
    print("Partial: \(partialAnswer)")
}
```

### FastVLMConfiguration

Configuration class for model inference settings.

#### Constructor

```swift
let config = FastVLMConfiguration()
config.computeUnits = .all
config.batchSize = 4
config.maxSequenceLength = 77
```

**Properties:**
- `computeUnits`: Core ML compute units (.all, .cpuAndGPU, .cpuOnly)
- `batchSize`: Batch size for inference (1-8)
- `maxSequenceLength`: Maximum text sequence length
- `temperature`: Sampling temperature for generation (0.0-2.0)
- `topP`: Top-p sampling parameter (0.0-1.0)

### FastVLMProfiler

Performance profiling utilities.

#### Methods

##### `profile(model: FastVLM, iterations: Int, warmup: Int) async throws -> ProfileMetrics`

Profile model performance over multiple iterations.

**Parameters:**
- `model`: FastVLM instance to profile
- `iterations`: Number of inference iterations
- `warmup`: Number of warmup iterations (excluded from metrics)

**Returns:**
- `ProfileMetrics`: Performance metrics

**Example:**
```swift
let profiler = FastVLMProfiler()
let metrics = try await profiler.profile(
    model: vlm,
    iterations: 100,
    warmup: 10
)
print("Average latency: \(metrics.avgLatencyMs)ms")
```

### ProfileMetrics

Performance metrics structure.

**Properties:**
- `avgLatencyMs`: Average inference latency in milliseconds
- `minLatencyMs`: Minimum latency
- `maxLatencyMs`: Maximum latency
- `p95LatencyMs`: 95th percentile latency
- `peakMemoryMB`: Peak memory usage in MB
- `energyImpact`: Energy impact score (1-5)

## Error Handling

### Python Exceptions

#### `FastVLMError`

Base exception class for Fast VLM errors.

#### `ModelLoadError`

Raised when model loading fails.

```python
try:
    model = converter.load_pytorch_model("invalid_path.pth")
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
```

#### `ConversionError`

Raised when Core ML conversion fails.

```python
try:
    coreml_model = converter.convert_to_coreml(model)
except ConversionError as e:
    print(f"Conversion failed: {e}")
```

### Swift Errors

#### `FastVLMError`

Enumeration of possible errors in Swift API.

```swift
enum FastVLMError: Error {
    case modelLoadFailed(String)
    case unsupportedDevice
    case inferenceError(String)
    case invalidInput
    case outOfMemory
    case timeout
}
```

**Error Handling Example:**
```swift
do {
    let answer = try await vlm.answer(image: image, question: question)
    print(answer)
} catch FastVLMError.modelLoadFailed(let message) {
    print("Model load failed: \(message)")
} catch FastVLMError.inferenceError(let message) {
    print("Inference error: \(message)")
} catch {
    print("Unexpected error: \(error)")
}
```

## Configuration

### Environment Variables

#### Python Configuration

- `FAST_VLM_MODEL_CACHE_DIR`: Directory for cached models (default: ~/.cache/fast_vlm)
- `FAST_VLM_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `FAST_VLM_DEVICE`: Preferred device (cpu, gpu, auto)
- `FAST_VLM_BATCH_SIZE`: Default batch size for processing

#### Example:
```bash
export FAST_VLM_MODEL_CACHE_DIR="/tmp/models"
export FAST_VLM_LOG_LEVEL="DEBUG"
export FAST_VLM_DEVICE="gpu"
```

### Configuration Files

#### `fast_vlm_config.yaml`

```yaml
models:
  cache_dir: "~/.cache/fast_vlm"
  default_quantization: "int4"
  
conversion:
  batch_size: 8
  compute_units: "ALL"
  
inference:
  timeout_seconds: 30
  max_memory_mb: 2048
  
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Examples

### Complete Python Example

```python
import numpy as np
from PIL import Image
from fast_vlm_ondevice import FastVLMConverter
from fast_vlm_ondevice.quantization import QuantizationConfig

# Initialize converter
converter = FastVLMConverter()

# Load model
model = converter.load_pytorch_model("checkpoints/fast-vlm-base.pth")

# Configure quantization
config = QuantizationConfig(
    vision_encoder="int4",
    text_encoder="int8",
    fusion_layers="fp16",
    decoder="int4"
)

# Convert to Core ML
coreml_model = converter.convert_to_coreml(
    model,
    quantization_config=config,
    image_size=(336, 336)
)

# Save model
coreml_model.save("FastVLM.mlpackage")
print(f"Model size: {converter.get_model_size_mb():.1f}MB")
```

### Complete Swift Example

```swift
import SwiftUI
import FastVLMKit

struct ContentView: View {
    @StateObject private var vlmManager = FastVLMManager()
    @State private var selectedImage: UIImage?
    @State private var question = ""
    @State private var answer = ""
    @State private var isProcessing = false
    
    var body: some View {
        VStack(spacing: 20) {
            // Image selection
            if let image = selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 300)
            } else {
                Button("Select Image") {
                    // Image picker logic
                }
            }
            
            // Question input
            TextField("Ask a question about the image...", text: $question)
                .textFieldStyle(RoundedBorderTextFieldStyle())
            
            // Process button
            Button("Get Answer") {
                processQuestion()
            }
            .disabled(selectedImage == nil || question.isEmpty || isProcessing)
            
            // Answer display
            if !answer.isEmpty {
                Text(answer)
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
            }
            
            if isProcessing {
                ProgressView("Processing...")
            }
        }
        .padding()
    }
    
    private func processQuestion() {
        guard let image = selectedImage else { return }
        
        isProcessing = true
        Task {
            do {
                let result = try await vlmManager.processQuery(
                    image: image,
                    question: question
                )
                await MainActor.run {
                    answer = result
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    answer = "Error: \(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }
}

@MainActor
class FastVLMManager: ObservableObject {
    private var vlm: FastVLM?
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        do {
            vlm = try FastVLM(modelPath: "FastVLM.mlpackage")
        } catch {
            print("Failed to load model: \(error)")
        }
    }
    
    func processQuery(image: UIImage, question: String) async throws -> String {
        guard let vlm = vlm else {
            throw FastVLMError.modelLoadFailed("Model not loaded")
        }
        
        return try await vlm.answer(image: image, question: question)
    }
}
```

## Best Practices

### Performance Optimization

1. **Batch Processing**: Use batch methods when processing multiple inputs
2. **Memory Management**: Monitor memory usage and clean up resources
3. **Caching**: Cache converted models to avoid repeated conversion
4. **Compute Units**: Use appropriate compute units for your hardware

### Error Handling

1. **Graceful Degradation**: Provide fallback behavior for errors
2. **User Feedback**: Show meaningful error messages to users
3. **Logging**: Log errors for debugging and monitoring
4. **Timeout Handling**: Set appropriate timeouts for inference

### Security

1. **Input Validation**: Validate all inputs before processing
2. **Resource Limits**: Set limits on memory and processing time
3. **Secure Storage**: Store models and sensitive data securely
4. **Privacy**: Ensure user data is processed on-device only

This API reference provides comprehensive documentation for both Python and Swift APIs, including examples, error handling, and best practices.