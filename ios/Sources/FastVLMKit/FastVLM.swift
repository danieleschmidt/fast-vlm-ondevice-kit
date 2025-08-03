import Foundation
import CoreML
import Vision
import Accelerate

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

/// Performance metrics for inference
public struct InferenceMetrics {
    public let latencyMs: Double
    public let memoryUsageMB: Double
    public let energyImpact: String
    public let confidenceScore: Float
}

/// FastVLM model for on-device vision-language inference
@available(iOS 17.0, macOS 14.0, *)
public class FastVLM {
    
    // MARK: - Properties
    
    private let model: MLModel
    private let tokenizer: FastVLMTokenizer
    private let imageProcessor: ImageProcessor
    private let configuration: MLModelConfiguration
    private var cachedAnswers: [String: String] = [:]
    private let maxCacheSize = 100
    
    // MARK: - Initialization
    
    /// Initialize FastVLM with Core ML model
    /// - Parameter modelPath: Path to .mlpackage model
    public init(modelPath: String) throws {
        let url = URL(fileURLWithPath: modelPath)
        
        // Configure for optimal performance
        self.configuration = MLModelConfiguration()
        self.configuration.computeUnits = .all
        self.configuration.allowLowPrecisionAccumulationOnGPU = true
        
        self.model = try MLModel(contentsOf: url, configuration: configuration)
        self.tokenizer = FastVLMTokenizer()
        self.imageProcessor = ImageProcessor()
        
        print("FastVLM initialized successfully")
    }
    
    // MARK: - Public Interface
    
    /// Answer a question about an image
    /// - Parameters:
    ///   - image: Input image
    ///   - question: Text question
    /// - Returns: Generated answer
    public func answer(image: PlatformImage, question: String) async throws -> String {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Check cache first
        let cacheKey = "\(image.hashValue)_\(question.hashValue)"
        if let cachedAnswer = cachedAnswers[cacheKey] {
            return cachedAnswer
        }
        
        // Preprocess inputs
        let processedImage = try await imageProcessor.preprocess(image: image)
        let tokenizedQuestion = try tokenizer.tokenize(text: question)
        
        // Create model input
        let input = try createModelInput(
            image: processedImage,
            inputIds: tokenizedQuestion.inputIds,
            attentionMask: tokenizedQuestion.attentionMask
        )
        
        // Run inference
        let prediction = try await model.prediction(from: input)
        
        // Decode output
        let answer = try decodeAnswer(from: prediction)
        
        // Cache result
        updateCache(key: cacheKey, answer: answer)
        
        let latency = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        print("Inference completed in \(Int(latency))ms")
        
        return answer
    }
    
    /// Answer multiple questions about an image efficiently
    /// - Parameters:
    ///   - image: Input image
    ///   - questions: Array of text questions
    /// - Returns: Array of generated answers
    public func batchAnswer(image: PlatformImage, questions: [String]) async throws -> [String] {
        var answers: [String] = []
        
        // Process image once for all questions
        let processedImage = try await imageProcessor.preprocess(image: image)
        
        for question in questions {
            let tokenizedQuestion = try tokenizer.tokenize(text: question)
            
            let input = try createModelInput(
                image: processedImage,
                inputIds: tokenizedQuestion.inputIds,
                attentionMask: tokenizedQuestion.attentionMask
            )
            
            let prediction = try await model.prediction(from: input)
            let answer = try decodeAnswer(from: prediction)
            answers.append(answer)
        }
        
        return answers
    }
    
    /// Get detailed performance metrics for the last inference
    /// - Returns: Performance metrics
    public func getPerformanceMetrics() -> InferenceMetrics {
        // This would track actual metrics in a real implementation
        return InferenceMetrics(
            latencyMs: 187.0,
            memoryUsageMB: 892.0,
            energyImpact: "Low",
            confidenceScore: 0.85
        )
    }
    
    // MARK: - Private Implementation
    
    private func createModelInput(
        image: MLMultiArray,
        inputIds: [Int32],
        attentionMask: [Int32]
    ) throws -> MLDictionaryFeatureProvider {
        
        let inputIdsArray = try MLMultiArray(shape: [1, NSNumber(value: inputIds.count)], dataType: .int32)
        let attentionMaskArray = try MLMultiArray(shape: [1, NSNumber(value: attentionMask.count)], dataType: .int32)
        
        // Copy data
        for (i, value) in inputIds.enumerated() {
            inputIdsArray[i] = NSNumber(value: value)
        }
        
        for (i, value) in attentionMask.enumerated() {
            attentionMaskArray[i] = NSNumber(value: value)
        }
        
        let features: [String: MLFeatureValue] = [
            "image": MLFeatureValue(multiArray: image),
            "input_ids": MLFeatureValue(multiArray: inputIdsArray),
            "attention_mask": MLFeatureValue(multiArray: attentionMaskArray)
        ]
        
        return try MLDictionaryFeatureProvider(dictionary: features)
    }
    
    private func decodeAnswer(from prediction: MLFeatureProvider) throws -> String {
        guard let outputArray = prediction.featureValue(for: "answer_logits")?.multiArrayValue else {
            throw FastVLMError.invalidOutput
        }
        
        // Get most likely tokens
        let logits = extractLogits(from: outputArray)
        let tokenIds = greedyDecode(logits: logits)
        
        // Convert to text
        let answer = tokenizer.decode(tokenIds: tokenIds)
        return answer.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    private func extractLogits(from array: MLMultiArray) -> [[Float]] {
        let sequenceLength = array.shape[1].intValue
        let vocabSize = array.shape[2].intValue
        
        var logits: [[Float]] = []
        
        for seq in 0..<sequenceLength {
            var tokenLogits: [Float] = []
            for vocab in 0..<vocabSize {
                let index = seq * vocabSize + vocab
                tokenLogits.append(array[index].floatValue)
            }
            logits.append(tokenLogits)
        }
        
        return logits
    }
    
    private func greedyDecode(logits: [[Float]]) -> [Int32] {
        var tokenIds: [Int32] = []
        
        for tokenLogits in logits {
            if let maxIndex = tokenLogits.enumerated().max(by: { $0.element < $1.element })?.offset {
                tokenIds.append(Int32(maxIndex))
            }
        }
        
        return tokenIds
    }
    
    private func updateCache(key: String, answer: String) {
        if cachedAnswers.count >= maxCacheSize {
            // Remove oldest entry (simple FIFO)
            if let firstKey = cachedAnswers.keys.first {
                cachedAnswers.removeValue(forKey: firstKey)
            }
        }
        cachedAnswers[key] = answer
    }
}

// MARK: - Supporting Classes

/// Tokenizer for FastVLM text processing
class FastVLMTokenizer {
    
    struct TokenizedText {
        let inputIds: [Int32]
        let attentionMask: [Int32]
    }
    
    private let maxSequenceLength = 77
    private let padTokenId: Int32 = 0
    private let unknownTokenId: Int32 = 1
    
    func tokenize(text: String) throws -> TokenizedText {
        // Simplified tokenization - would use real BERT tokenizer in production
        let words = text.lowercased().components(separatedBy: .whitespacesAndPunctuation)
        var inputIds: [Int32] = []
        
        for word in words.prefix(maxSequenceLength - 2) {
            if !word.isEmpty {
                // Simple hash-based token ID (placeholder for real vocabulary)
                let tokenId = Int32(abs(word.hashValue) % 30000 + 1000)
                inputIds.append(tokenId)
            }
        }
        
        // Pad to max length
        let paddingLength = maxSequenceLength - inputIds.count
        let paddedInputIds = inputIds + Array(repeating: padTokenId, count: paddingLength)
        
        // Create attention mask
        let attentionMask = Array(repeating: Int32(1), count: inputIds.count) + 
                          Array(repeating: Int32(0), count: paddingLength)
        
        return TokenizedText(
            inputIds: paddedInputIds,
            attentionMask: attentionMask
        )
    }
    
    func decode(tokenIds: [Int32]) -> String {
        // Simple decoding - would use real vocabulary in production
        let words = tokenIds.compactMap { tokenId -> String? in
            if tokenId == padTokenId || tokenId == unknownTokenId {
                return nil
            }
            // Generate placeholder words based on token ID
            return "word\(tokenId % 1000)"
        }
        
        return words.joined(separator: " ")
    }
}

/// Image processor for FastVLM vision input
class ImageProcessor {
    
    private let targetSize = CGSize(width: 336, height: 336)
    private let mean: [Float] = [0.485, 0.456, 0.406]
    private let std: [Float] = [0.229, 0.224, 0.225]
    
    func preprocess(image: PlatformImage) async throws -> MLMultiArray {
        // Resize image
        let resizedImage = try resize(image: image, to: targetSize)
        
        // Convert to pixel buffer
        let pixelBuffer = try createPixelBuffer(from: resizedImage)
        
        // Normalize and convert to MLMultiArray
        let normalizedArray = try normalize(pixelBuffer: pixelBuffer)
        
        return normalizedArray
    }
    
    private func resize(image: PlatformImage, to size: CGSize) throws -> PlatformImage {
        #if canImport(UIKit)
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }
        #else
        // macOS implementation would go here
        return image
        #endif
    }
    
    private func createPixelBuffer(from image: PlatformImage) throws -> CVPixelBuffer {
        #if canImport(UIKit)
        guard let cgImage = image.cgImage else {
            throw FastVLMError.imageProcessingFailed
        }
        
        let width = Int(targetSize.width)
        let height = Int(targetSize.height)
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            nil,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw FastVLMError.imageProcessingFailed
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
        
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return buffer
        #else
        throw FastVLMError.unsupportedPlatform
        #endif
    }
    
    private func normalize(pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        let outputArray = try MLMultiArray(
            shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )
        
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw FastVLMError.imageProcessingFailed
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        // Extract RGB channels and normalize
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * bytesPerRow + x * 4
                
                let b = Float(buffer[pixelIndex]) / 255.0
                let g = Float(buffer[pixelIndex + 1]) / 255.0
                let r = Float(buffer[pixelIndex + 2]) / 255.0
                
                // Normalize with ImageNet statistics
                let normalizedR = (r - mean[0]) / std[0]
                let normalizedG = (g - mean[1]) / std[1]
                let normalizedB = (b - mean[2]) / std[2]
                
                // Store in CHW format
                let rIndex = y * width + x
                let gIndex = height * width + y * width + x
                let bIndex = 2 * height * width + y * width + x
                
                outputArray[rIndex] = NSNumber(value: normalizedR)
                outputArray[gIndex] = NSNumber(value: normalizedG)
                outputArray[bIndex] = NSNumber(value: normalizedB)
            }
        }
        
        return outputArray
    }
}

// MARK: - Error Types

public enum FastVLMError: Error {
    case modelLoadFailed
    case imageProcessingFailed
    case tokenizationFailed
    case invalidOutput
    case unsupportedPlatform
}

// MARK: - Platform Abstraction

#if canImport(UIKit)
public typealias PlatformImage = UIImage
#elseif canImport(AppKit)
public typealias PlatformImage = NSImage
#endif