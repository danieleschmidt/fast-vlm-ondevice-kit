"""
Core pipeline for FastVLM on-device processing.

Implements the main processing pipeline with error recovery, 
monitoring, and optimization for mobile deployment.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import uuid
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages."""
    INITIALIZATION = "initialization"
    IMAGE_PREPROCESSING = "image_preprocessing" 
    TEXT_PREPROCESSING = "text_preprocessing"
    VISION_ENCODING = "vision_encoding"
    TEXT_ENCODING = "text_encoding"
    MULTIMODAL_FUSION = "multimodal_fusion"
    ANSWER_GENERATION = "answer_generation"
    POSTPROCESSING = "postprocessing"
    COMPLETE = "complete"


@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_latency_ms: Optional[float] = None
    stage_timings: Dict[str, float] = field(default_factory=dict)
    memory_usage_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    error_count: int = 0
    warnings_count: int = 0
    success: bool = False
    
    def complete_pipeline(self):
        """Mark pipeline as complete and calculate metrics."""
        self.end_time = time.time()
        self.total_latency_ms = (self.end_time - self.start_time) * 1000
        self.success = True
    
    def add_stage_timing(self, stage: PipelineStage, duration_ms: float):
        """Add timing for a specific stage."""
        self.stage_timings[stage.value] = duration_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "session_id": self.session_id,
            "total_latency_ms": self.total_latency_ms,
            "stage_timings": self.stage_timings,
            "memory_usage_mb": self.memory_usage_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "error_count": self.error_count,
            "warnings_count": self.warnings_count,
            "success": self.success
        }


@dataclass
class PipelineConfig:
    """Configuration for the FastVLM pipeline."""
    model_path: str
    image_size: Tuple[int, int] = (336, 336)
    max_sequence_length: int = 77
    temperature: float = 0.7
    top_k: int = 50
    max_new_tokens: int = 100
    batch_size: int = 1
    
    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_prefetch: bool = True
    parallel_processing: bool = True
    
    # Quality settings
    confidence_threshold: float = 0.5
    enable_safety_checks: bool = True
    enable_profiling: bool = False
    
    # Mobile optimization
    target_latency_ms: float = 250.0
    memory_limit_mb: float = 1500.0
    energy_efficient: bool = True


class FastVLMPipeline:
    """Main FastVLM processing pipeline optimized for mobile devices."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the FastVLM pipeline."""
        self.config = config
        self.metrics = PipelineMetrics()
        self.is_initialized = False
        self.cache = {}
        self._model = None
        self._tokenizer = None
        self._preprocessor = None
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.metrics.session_id[:8]}")
        
        # Initialize components
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Set up pipeline components."""
        self.logger.info(f"Initializing FastVLM pipeline (session: {self.metrics.session_id[:8]})")
        
        try:
            # Initialize model components (placeholder for actual implementation)
            self._initialize_model()
            self._initialize_preprocessors()
            self._initialize_cache()
            
            self.is_initialized = True
            self.logger.info("Pipeline initialization complete")
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {e}")
            self.metrics.error_count += 1
            raise
    
    def _initialize_model(self):
        """Initialize the Core ML model."""
        self.logger.info(f"Loading model from {self.config.model_path}")
        
        # Simulate model loading with error handling
        try:
            if not Path(self.config.model_path).exists():
                self.logger.warning(f"Model path does not exist: {self.config.model_path}")
                self.logger.info("Using demo/fallback model")
                self._model = {"type": "demo", "name": "FastVLM-Demo"}
            else:
                self._model = {"type": "coreml", "path": self.config.model_path}
                
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            self.metrics.error_count += 1
            # Fallback to demo model
            self._model = {"type": "demo", "name": "FastVLM-Demo"}
            self.metrics.warnings_count += 1
    
    def _initialize_preprocessors(self):
        """Initialize image and text preprocessors."""
        self.logger.info("Initializing preprocessors")
        
        # Image preprocessing setup
        self._preprocessor = {
            "image": {
                "target_size": self.config.image_size,
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225]
            },
            "text": {
                "max_length": self.config.max_sequence_length,
                "padding": True,
                "truncation": True
            }
        }
        
        # Initialize tokenizer (placeholder)
        self._tokenizer = {"vocab_size": 50000, "type": "clip"}
    
    def _initialize_cache(self):
        """Initialize caching system."""
        if self.config.enable_caching:
            self.logger.info(f"Initializing cache (size: {self.config.cache_size_mb}MB)")
            self.cache = {
                "vision_features": {},
                "text_embeddings": {},
                "fusion_cache": {},
                "max_size": self.config.cache_size_mb * 1024 * 1024  # Convert to bytes
            }
    
    @contextmanager
    def _stage_timer(self, stage: PipelineStage):
        """Context manager for timing pipeline stages."""
        start_time = time.time()
        self.logger.debug(f"Starting stage: {stage.value}")
        
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.add_stage_timing(stage, duration_ms)
            self.logger.debug(f"Completed stage {stage.value} in {duration_ms:.1f}ms")
    
    async def process(self, image_data: Any, question: str) -> Dict[str, Any]:
        """
        Process an image-question pair through the FastVLM pipeline.
        
        Args:
            image_data: Input image (PIL Image, numpy array, or path)
            question: Text question about the image
            
        Returns:
            Dictionary containing answer and processing metadata
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")
        
        self.logger.info(f"Processing query: '{question[:50]}...'")
        result = {
            "answer": "",
            "confidence": 0.0,
            "processing_time_ms": 0.0,
            "session_id": self.metrics.session_id,
            "error": None
        }
        
        try:
            # Stage 1: Image Preprocessing
            with self._stage_timer(PipelineStage.IMAGE_PREPROCESSING):
                image_features = await self._preprocess_image(image_data)
            
            # Stage 2: Text Preprocessing  
            with self._stage_timer(PipelineStage.TEXT_PREPROCESSING):
                text_tokens = await self._preprocess_text(question)
            
            # Stage 3: Vision Encoding
            with self._stage_timer(PipelineStage.VISION_ENCODING):
                vision_embeddings = await self._encode_vision(image_features)
            
            # Stage 4: Text Encoding
            with self._stage_timer(PipelineStage.TEXT_ENCODING):
                text_embeddings = await self._encode_text(text_tokens)
            
            # Stage 5: Multimodal Fusion
            with self._stage_timer(PipelineStage.MULTIMODAL_FUSION):
                fused_features = await self._fuse_modalities(vision_embeddings, text_embeddings)
            
            # Stage 6: Answer Generation
            with self._stage_timer(PipelineStage.ANSWER_GENERATION):
                answer_result = await self._generate_answer(fused_features, question)
            
            # Stage 7: Postprocessing
            with self._stage_timer(PipelineStage.POSTPROCESSING):
                final_result = await self._postprocess_answer(answer_result)
            
            result.update(final_result)
            self.metrics.complete_pipeline()
            
            self.logger.info(f"Processing complete - Answer: '{result['answer'][:50]}...'")
            self.logger.info(f"Total latency: {self.metrics.total_latency_ms:.1f}ms")
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            self.metrics.error_count += 1
            result["error"] = str(e)
            result["answer"] = "I'm sorry, I encountered an error processing your request."
        
        result["metrics"] = self.metrics.to_dict()
        return result
    
    async def _preprocess_image(self, image_data: Any) -> Dict[str, Any]:
        """Preprocess input image for vision encoder."""
        self.logger.debug("Preprocessing image")
        
        # Simulate image preprocessing
        await asyncio.sleep(0.005)  # 5ms processing time
        
        return {
            "tensor_shape": [1, 3] + list(self.config.image_size),
            "data_type": "float32",
            "preprocessing_applied": ["resize", "normalize", "to_tensor"]
        }
    
    async def _preprocess_text(self, question: str) -> Dict[str, Any]:
        """Preprocess input text for text encoder."""
        self.logger.debug(f"Preprocessing text: '{question[:30]}...'")
        
        # Simulate tokenization
        await asyncio.sleep(0.002)  # 2ms processing time
        
        # Simple word-based tokenization simulation
        tokens = question.lower().split()
        token_ids = [hash(token) % 50000 for token in tokens]
        
        return {
            "token_ids": token_ids[:self.config.max_sequence_length],
            "attention_mask": [1] * min(len(token_ids), self.config.max_sequence_length),
            "sequence_length": len(tokens)
        }
    
    async def _encode_vision(self, image_features: Dict[str, Any]) -> Dict[str, Any]:
        """Encode image features using vision encoder."""
        self.logger.debug("Encoding vision features")
        
        # Check cache first
        cache_key = f"vision_{hash(str(image_features))}"
        if self.config.enable_caching and cache_key in self.cache.get("vision_features", {}):
            self.logger.debug("Using cached vision features")
            return self.cache["vision_features"][cache_key]
        
        # Simulate vision encoding
        await asyncio.sleep(0.050)  # 50ms processing time
        
        vision_embeddings = {
            "embeddings": [0.1] * 768,  # Simulated embeddings
            "spatial_features": {"height": 24, "width": 24, "channels": 768},
            "global_features": {"dimension": 768}
        }
        
        # Cache results
        if self.config.enable_caching:
            self.cache["vision_features"][cache_key] = vision_embeddings
        
        return vision_embeddings
    
    async def _encode_text(self, text_tokens: Dict[str, Any]) -> Dict[str, Any]:
        """Encode text tokens using text encoder."""
        self.logger.debug("Encoding text features")
        
        # Check cache first
        cache_key = f"text_{hash(str(text_tokens['token_ids']))}"
        if self.config.enable_caching and cache_key in self.cache.get("text_embeddings", {}):
            self.logger.debug("Using cached text embeddings")
            return self.cache["text_embeddings"][cache_key]
        
        # Simulate text encoding
        await asyncio.sleep(0.020)  # 20ms processing time
        
        text_embeddings = {
            "embeddings": [0.05] * 512,  # Simulated embeddings
            "sequence_embeddings": [[0.05] * 512] * text_tokens["sequence_length"],
            "pooled_features": {"dimension": 512}
        }
        
        # Cache results
        if self.config.enable_caching:
            self.cache["text_embeddings"][cache_key] = text_embeddings
        
        return text_embeddings
    
    async def _fuse_modalities(self, vision_embeddings: Dict[str, Any], text_embeddings: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse vision and text modalities using cross-attention."""
        self.logger.debug("Performing multimodal fusion")
        
        # Simulate cross-modal attention
        await asyncio.sleep(0.080)  # 80ms processing time
        
        fused_features = {
            "cross_attention_features": [0.03] * 1024,
            "vision_weight": 0.6,
            "text_weight": 0.4,
            "fusion_confidence": 0.85
        }
        
        return fused_features
    
    async def _generate_answer(self, fused_features: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Generate answer using the decoder."""
        self.logger.debug("Generating answer")
        
        # Simulate answer generation
        await asyncio.sleep(0.090)  # 90ms processing time
        
        # Simple answer generation simulation
        sample_answers = [
            "I can see a person in the image.",
            "There are several objects including a car and a building.",
            "The image shows a beautiful landscape with mountains.",
            "I can identify multiple items in this scene.",
            "The image contains both indoor and outdoor elements."
        ]
        
        answer = sample_answers[hash(question) % len(sample_answers)]
        
        return {
            "answer": answer,
            "confidence": min(0.95, fused_features.get("fusion_confidence", 0.8) + 0.1),
            "generation_steps": 15,
            "alternative_answers": sample_answers[:3]
        }
    
    async def _postprocess_answer(self, answer_result: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess generated answer."""
        self.logger.debug("Postprocessing answer")
        
        # Simulate postprocessing
        await asyncio.sleep(0.005)  # 5ms processing time
        
        # Apply safety checks and confidence filtering
        if answer_result["confidence"] < self.config.confidence_threshold:
            self.logger.warning(f"Low confidence answer: {answer_result['confidence']}")
            self.metrics.warnings_count += 1
        
        return {
            "answer": answer_result["answer"].strip(),
            "confidence": answer_result["confidence"],
            "safe": True,  # Placeholder for safety check
            "postprocessed": True
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        return {
            "pipeline_metrics": self.metrics.to_dict(),
            "cache_stats": self._get_cache_stats(),
            "model_info": {
                "type": self._model.get("type", "unknown"),
                "path": self._model.get("path", ""),
                "initialized": self.is_initialized
            },
            "config": {
                "image_size": self.config.image_size,
                "max_sequence_length": self.config.max_sequence_length,
                "target_latency_ms": self.config.target_latency_ms,
                "memory_limit_mb": self.config.memory_limit_mb
            }
        }
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache utilization statistics."""
        if not self.config.enable_caching:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "vision_cache_size": len(self.cache.get("vision_features", {})),
            "text_cache_size": len(self.cache.get("text_embeddings", {})),
            "fusion_cache_size": len(self.cache.get("fusion_cache", {})),
            "max_size_mb": self.config.cache_size_mb
        }
    
    def clear_cache(self):
        """Clear all caches to free memory."""
        self.logger.info("Clearing pipeline caches")
        if hasattr(self, 'cache'):
            self.cache.clear()
    
    def __del__(self):
        """Cleanup resources when pipeline is destroyed."""
        if hasattr(self, 'logger'):
            self.logger.info(f"Pipeline cleanup (session: {self.metrics.session_id[:8]})")