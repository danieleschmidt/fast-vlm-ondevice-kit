"""
Core pipeline implementation for FastVLM inference without external dependencies.

This module provides a lightweight implementation that demonstrates the architecture
and can run without PyTorch, CoreML, or other heavy dependencies.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import base64

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for VLM inference."""
    model_name: str = "fast-vlm-base"
    max_sequence_length: int = 77
    image_size: Tuple[int, int] = (336, 336)
    quantization_bits: int = 4
    enable_caching: bool = True
    timeout_seconds: float = 30.0
    batch_size: int = 1


@dataclass
class InferenceResult:
    """Result from VLM inference."""
    answer: str
    confidence: float
    latency_ms: float
    model_used: str
    timestamp: str
    metadata: Dict[str, Any]


class MockVisionEncoder:
    """Mock vision encoder for demonstration purposes."""
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.feature_dim = {"tiny": 256, "base": 512, "large": 768}[model_size]
        logger.info(f"Initialized MockVisionEncoder ({model_size}, {self.feature_dim}d)")
    
    def encode_image(self, image_data: bytes) -> Dict[str, Any]:
        """Encode image to feature representation."""
        # Simulate image encoding with deterministic features
        import hashlib
        image_hash = hashlib.md5(image_data).hexdigest()
        
        # Generate mock features based on image hash (deterministic)
        features = [
            (int(image_hash[i:i+2], 16) / 255.0 - 0.5) * 2  # Normalize to [-1, 1]
            for i in range(0, min(len(image_hash), self.feature_dim * 2), 2)
        ]
        
        # Pad or truncate to feature_dim
        if len(features) < self.feature_dim:
            features.extend([0.0] * (self.feature_dim - len(features)))
        else:
            features = features[:self.feature_dim]
        
        return {
            "features": features,
            "spatial_features": [features[:64], features[64:128]] if len(features) >= 128 else [features],
            "attention_mask": [1.0] * min(64, len(features)),
            "image_hash": image_hash
        }


class MockTextEncoder:
    """Mock text encoder for demonstration purposes."""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.embedding_dim = 512
        logger.info(f"Initialized MockTextEncoder (vocab={vocab_size}, dim={self.embedding_dim})")
    
    def encode_text(self, text: str) -> Dict[str, Any]:
        """Encode text to feature representation."""
        # Simple tokenization simulation
        tokens = text.lower().split()
        
        # Generate mock embeddings based on text content
        text_features = []
        for token in tokens[:20]:  # Max 20 tokens
            token_hash = hash(token) % 1000
            embedding = [(token_hash + i) / 1000.0 for i in range(self.embedding_dim)]
            text_features.append(embedding)
        
        # Pad to fixed length
        max_length = 20
        while len(text_features) < max_length:
            text_features.append([0.0] * self.embedding_dim)
        
        return {
            "token_embeddings": text_features[:max_length],
            "attention_mask": [1.0] * min(len(tokens), max_length) + [0.0] * max(0, max_length - len(tokens)),
            "sequence_length": min(len(tokens), max_length),
            "original_text": text
        }


class MockFusionModule:
    """Mock cross-modal fusion module."""
    
    def __init__(self, vision_dim: int = 512, text_dim: int = 512):
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.fusion_dim = 768
        logger.info(f"Initialized MockFusionModule (vision={vision_dim}, text={text_dim}, fusion={self.fusion_dim})")
    
    def fuse_modalities(self, vision_features: Dict[str, Any], text_features: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse vision and text features."""
        # Simulate cross-attention fusion
        v_features = vision_features["features"]
        t_features = text_features["token_embeddings"][0]  # Use first token embedding
        
        # Simple fusion: concatenate and project
        fused_features = []
        for i in range(min(len(v_features), len(t_features))):
            fused_val = (v_features[i] + t_features[i]) / 2.0
            fused_features.append(fused_val)
        
        # Simulate attention scores
        attention_scores = [abs(f) for f in fused_features[:10]]
        
        return {
            "fused_features": fused_features,
            "cross_attention": attention_scores,
            "fusion_quality": sum(attention_scores) / len(attention_scores) if attention_scores else 0.0
        }


class MockAnswerGenerator:
    """Mock answer generation module."""
    
    def __init__(self):
        self.common_responses = {
            "what": ["objects", "items", "things", "elements"],
            "how": ["process", "method", "way", "approach"], 
            "where": ["location", "place", "position", "area"],
            "when": ["time", "moment", "period", "duration"],
            "who": ["person", "people", "individual", "character"],
            "why": ["reason", "cause", "purpose", "explanation"]
        }
        logger.info("Initialized MockAnswerGenerator")
    
    def generate_answer(self, fused_features: Dict[str, Any], question: str) -> str:
        """Generate answer based on fused features and question."""
        question_lower = question.lower()
        
        # Simple rule-based generation
        if any(word in question_lower for word in ["what", "object", "thing"]):
            confidence = fused_features.get("fusion_quality", 0.5)
            if confidence > 0.7:
                return "I can see several objects including furniture, decorative items, and possibly electronic devices in the image."
            elif confidence > 0.4:
                return "There appear to be some objects in the image, but I need better image quality for specific identification."
            else:
                return "I can detect some items but cannot clearly identify specific objects."
        
        elif any(word in question_lower for word in ["color", "colour"]):
            return "The image contains various colors including what appears to be neutral tones with some accent colors."
        
        elif any(word in question_lower for word in ["person", "people", "human"]):
            return "I can analyze the image for people, but I prioritize privacy and provide general descriptions only."
        
        elif any(word in question_lower for word in ["count", "how many", "number"]):
            attention_scores = fused_features.get("cross_attention", [])
            count = len([s for s in attention_scores if s > 0.5])
            return f"Based on my analysis, I can identify approximately {count} distinct elements or regions of interest."
        
        elif any(word in question_lower for word in ["describe", "scene", "setting"]):
            return "This appears to be an indoor scene with various objects and elements arranged in what looks like a lived-in space."
        
        else:
            return "I can see various elements in the image. Could you ask a more specific question about what you'd like to know?"


class FastVLMCorePipeline:
    """Core FastVLM inference pipeline without external dependencies."""
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """Initialize the core pipeline."""
        self.config = config or InferenceConfig()
        
        # Initialize mock components
        model_size = self._determine_model_size()
        self.vision_encoder = MockVisionEncoder(model_size)
        self.text_encoder = MockTextEncoder()
        self.fusion_module = MockFusionModule()
        self.answer_generator = MockAnswerGenerator()
        
        # Simple cache
        self.cache = {} if self.config.enable_caching else None
        
        logger.info(f"Initialized FastVLMCorePipeline with {self.config.model_name}")
    
    def _determine_model_size(self) -> str:
        """Determine model size from config."""
        if "tiny" in self.config.model_name:
            return "tiny"
        elif "large" in self.config.model_name:
            return "large"
        else:
            return "base"
    
    def _generate_cache_key(self, image_data: bytes, question: str) -> str:
        """Generate cache key for image-question pair."""
        import hashlib
        combined = base64.b64encode(image_data).decode() + question
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def process_image_question(self, image_data: bytes, question: str) -> InferenceResult:
        """Process image and question to generate answer."""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(image_data, question) if self.cache is not None else None
        if cache_key and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            logger.info(f"Cache hit for key {cache_key}")
            return InferenceResult(**cached_result)
        
        try:
            # Step 1: Encode image
            vision_features = self.vision_encoder.encode_image(image_data)
            
            # Step 2: Encode text
            text_features = self.text_encoder.encode_text(question)
            
            # Step 3: Fuse modalities
            fused_features = self.fusion_module.fuse_modalities(vision_features, text_features)
            
            # Step 4: Generate answer
            answer = self.answer_generator.generate_answer(fused_features, question)
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            confidence = fused_features.get("fusion_quality", 0.5)
            
            # Create result
            result = InferenceResult(
                answer=answer,
                confidence=confidence,
                latency_ms=latency_ms,
                model_used=self.config.model_name,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                metadata={
                    "vision_features_dim": len(vision_features["features"]),
                    "text_tokens": text_features["sequence_length"],
                    "fusion_dim": len(fused_features["fused_features"]),
                    "image_hash": vision_features["image_hash"],
                    "cache_used": False
                }
            )
            
            # Cache result
            if cache_key and self.cache is not None:
                self.cache[cache_key] = asdict(result)
                logger.info(f"Cached result for key {cache_key}")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return InferenceResult(
                answer=f"Error processing request: {str(e)}",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                model_used=self.config.model_name,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                metadata={"error": str(e)}
            )
    
    def process_text_only(self, question: str) -> InferenceResult:
        """Process text-only question (no image)."""
        start_time = time.time()
        
        try:
            # Create dummy image data for consistency
            dummy_image = b"dummy_image_data"
            
            # Use general knowledge responses
            answer = self._generate_general_response(question)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return InferenceResult(
                answer=answer,
                confidence=0.8,  # High confidence for general knowledge
                latency_ms=latency_ms,
                model_used=self.config.model_name,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                metadata={
                    "mode": "text_only",
                    "question_length": len(question),
                    "response_type": "general_knowledge"
                }
            )
            
        except Exception as e:
            logger.error(f"Text-only processing failed: {e}")
            return InferenceResult(
                answer=f"Error processing text question: {str(e)}",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                model_used=self.config.model_name,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                metadata={"error": str(e)}
            )
    
    def _generate_general_response(self, question: str) -> str:
        """Generate general knowledge response."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["hello", "hi", "greeting"]):
            return "Hello! I'm FastVLM, a vision-language model. I can analyze images and answer questions about them."
        
        elif any(word in question_lower for word in ["help", "what can you do"]):
            return "I can analyze images and answer questions about what I see, including objects, scenes, colors, and spatial relationships."
        
        elif any(word in question_lower for word in ["fastVLM", "fast vlm", "model"]):
            return "FastVLM is an efficient vision-language model optimized for mobile devices, capable of <250ms inference on modern smartphones."
        
        else:
            return "I'm designed to analyze images and answer questions about them. Please provide an image along with your question for the best results."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        cache_size = len(self.cache) if self.cache else 0
        
        return {
            "model_name": self.config.model_name,
            "cache_enabled": self.config.enable_caching,
            "cache_entries": cache_size,
            "quantization_bits": self.config.quantization_bits,
            "max_sequence_length": self.config.max_sequence_length,
            "image_size": self.config.image_size,
            "components": {
                "vision_encoder": self.vision_encoder.model_size,
                "text_encoder": f"{self.text_encoder.vocab_size} vocab",
                "fusion_module": f"{self.fusion_module.fusion_dim}d",
                "answer_generator": "rule_based"
            }
        }
    
    def clear_cache(self) -> int:
        """Clear inference cache and return number of entries cleared."""
        if self.cache:
            entries = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared {entries} cache entries")
            return entries
        return 0


# Convenience function for quick inference
def quick_inference(image_data: bytes, question: str, model_name: str = "fast-vlm-base") -> Dict[str, Any]:
    """Quick inference function for simple use cases."""
    config = InferenceConfig(model_name=model_name)
    pipeline = FastVLMCorePipeline(config)
    result = pipeline.process_image_question(image_data, question)
    return asdict(result)


# Demo data generator
def create_demo_image() -> bytes:
    """Create demo image data for testing."""
    # Generate deterministic "image" data
    demo_data = "demo_image_" + "x" * 100  # Simulate image bytes
    return demo_data.encode()


if __name__ == "__main__":
    # Demo usage
    print("FastVLM Core Pipeline Demo")
    print("=" * 30)
    
    # Create pipeline
    config = InferenceConfig(model_name="fast-vlm-base", enable_caching=True)
    pipeline = FastVLMCorePipeline(config)
    
    # Demo image and questions
    demo_image = create_demo_image()
    demo_questions = [
        "What objects are in this image?",
        "What colors do you see?",
        "Describe the scene",
        "How many items are visible?"
    ]
    
    print(f"\nProcessing {len(demo_questions)} questions...")
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{i}. Question: {question}")
        result = pipeline.process_image_question(demo_image, question)
        print(f"   Answer: {result.answer}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Latency: {result.latency_ms:.1f}ms")
    
    # Show stats
    print(f"\nPipeline Stats:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")