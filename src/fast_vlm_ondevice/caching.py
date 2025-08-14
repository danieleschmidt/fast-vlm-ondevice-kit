"""
Advanced caching system for FastVLM models.

Provides intelligent caching with TTL, LRU eviction, and persistence.
"""

import time
import json
import hashlib
import threading
import pickle
from typing import Dict, Any, Optional, Union, Callable, Tuple
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, asdict
import logging

try:
    import numpy as np
    from PIL import Image
    CACHE_DEPS = True
except ImportError:
    CACHE_DEPS = False
    # Create fallback classes for missing dependencies
    class np:
        class ndarray: pass
    class Image:
        class Image: pass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int
    last_access: float
    ttl: float
    size_bytes: int
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """Update access timestamp and count."""
        self.last_access = time.time()
        self.access_count += 1


class LRUCache:
    """Thread-safe LRU cache with TTL and size limits."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 512.0,
        default_ttl: float = 3600.0,
        cleanup_interval: float = 300.0
    ):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default TTL in seconds
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
            "current_size": 0,
            "current_memory": 0
        }
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._running = True
        self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats["misses"] += 1
                return None
            
            if entry.is_expired():
                self._remove_entry(key)
                self._stats["expired"] += 1
                self._stats["misses"] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats["hits"] += 1
            
            return entry.value
    
    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> bool:
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (None for default)
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            ttl = ttl or self.default_ttl
            size_bytes = self._estimate_size(value)
            
            # Check if value is too large
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                access_count=1,
                last_access=time.time(),
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # Ensure space
            self._ensure_space(size_bytes)
            
            # Add entry
            self._cache[key] = entry
            self._stats["current_size"] += 1
            self._stats["current_memory"] += size_bytes
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats["current_size"] = 0
            self._stats["current_memory"] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                **self._stats.copy(),
                "hit_rate": hit_rate,
                "memory_usage_mb": self._stats["current_memory"] / (1024 * 1024),
                "memory_usage_percent": (self._stats["current_memory"] / self.max_memory_bytes) * 100
            }
    
    def _ensure_space(self, needed_bytes: int):
        """Ensure enough space for new entry."""
        while (
            (self._stats["current_size"] >= self.max_size) or
            (self._stats["current_memory"] + needed_bytes > self.max_memory_bytes)
        ):
            if not self._cache:
                break
                
            # Remove least recently used entry
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._stats["evictions"] += 1
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        entry = self._cache.pop(key, None)
        if entry:
            self._stats["current_size"] -= 1
            self._stats["current_memory"] -= entry.size_bytes
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return len(json.dumps(value).encode())
            elif hasattr(value, 'nbytes'):  # NumPy array
                return value.nbytes
            else:
                # Fallback to pickle size
                return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
    
    def _cleanup_loop(self):
        """Background cleanup of expired entries."""
        while self._running:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
                self._stats["expired"] += 1
    
    def shutdown(self):
        """Shutdown cache and cleanup thread."""
        self._running = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)


class ModelCache:
    """Specialized cache for ML models."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_models: int = 10,
        max_memory_gb: float = 4.0
    ):
        """Initialize model cache.
        
        Args:
            cache_dir: Directory for persistent cache
            max_models: Maximum number of cached models
            max_memory_gb: Maximum memory usage in GB
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_cache = LRUCache(
            max_size=max_models,
            max_memory_mb=max_memory_gb * 1024,
            default_ttl=7200.0  # 2 hours
        )
        
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get cached model."""
        # Try memory cache first
        model = self.memory_cache.get(model_id)
        if model is not None:
            logger.debug(f"Model cache hit (memory): {model_id}")
            return model
        
        # Try disk cache
        if self.cache_dir:
            model = self._load_from_disk(model_id)
            if model is not None:
                logger.debug(f"Model cache hit (disk): {model_id}")
                # Put back in memory cache
                self.memory_cache.put(model_id, model)
                return model
        
        logger.debug(f"Model cache miss: {model_id}")
        return None
    
    def put_model(
        self,
        model_id: str,
        model: Any,
        metadata: Optional[Dict[str, Any]] = None,
        persist: bool = True
    ) -> bool:
        """Cache model.
        
        Args:
            model_id: Unique model identifier
            model: Model object to cache
            metadata: Model metadata
            persist: Whether to persist to disk
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            # Store metadata
            self._model_metadata[model_id] = metadata or {}
            self._model_metadata[model_id]["cached_at"] = time.time()
            
            # Cache in memory
            success = self.memory_cache.put(model_id, model)
            
            # Persist to disk if requested
            if success and persist and self.cache_dir:
                try:
                    self._save_to_disk(model_id, model, metadata)
                except Exception as e:
                    logger.warning(f"Failed to persist model to disk: {e}")
            
            return success
    
    def _save_to_disk(self, model_id: str, model: Any, metadata: Optional[Dict[str, Any]]):
        """Save model to disk."""
        model_dir = self.cache_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata or {}, f, indent=2)
        
        # Save model (method depends on model type)
        model_file = model_dir / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
    
    def _load_from_disk(self, model_id: str) -> Optional[Any]:
        """Load model from disk."""
        if not self.cache_dir:
            return None
        
        model_dir = self.cache_dir / model_id
        if not model_dir.exists():
            return None
        
        try:
            # Load model
            model_file = model_dir / "model.pkl"
            if not model_file.exists():
                return None
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                self._model_metadata[model_id] = metadata
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from disk: {e}")
            return None
    
    def invalidate_model(self, model_id: str):
        """Invalidate cached model."""
        self.memory_cache.delete(model_id)
        
        if model_id in self._model_metadata:
            del self._model_metadata[model_id]
        
        # Remove from disk
        if self.cache_dir:
            model_dir = self.cache_dir / model_id
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        
        disk_stats = {"models": 0, "size_mb": 0.0}
        if self.cache_dir and self.cache_dir.exists():
            for model_dir in self.cache_dir.iterdir():
                if model_dir.is_dir():
                    disk_stats["models"] += 1
                    for file in model_dir.rglob("*"):
                        if file.is_file():
                            disk_stats["size_mb"] += file.stat().st_size / (1024 * 1024)
        
        return {
            "memory": memory_stats,
            "disk": disk_stats,
            "metadata_entries": len(self._model_metadata)
        }


class InferenceCache:
    """Cache for inference results."""
    
    def __init__(self, max_size: int = 10000, ttl: float = 1800.0):
        """Initialize inference cache.
        
        Args:
            max_size: Maximum number of cached results
            ttl: Time to live in seconds (30 minutes default)
        """
        self.cache = LRUCache(
            max_size=max_size,
            max_memory_mb=1024.0,  # 1GB
            default_ttl=ttl
        )
    
    def get_cache_key(
        self,
        image_hash: str,
        question: str,
        model_id: str,
        model_version: str = "1.0"
    ) -> str:
        """Generate cache key for inference."""
        key_data = f"{model_id}:{model_version}:{image_hash}:{question}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get_image_hash(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        """Generate hash for image."""
        try:
            if isinstance(image, str):
                # File path
                with open(image, 'rb') as f:
                    image_data = f.read()
            elif isinstance(image, Image.Image):
                # PIL Image
                import io
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                image_data = buffer.getvalue()
            elif isinstance(image, np.ndarray):
                # NumPy array
                image_data = image.tobytes()
            else:
                # Fallback
                image_data = str(image).encode()
            
            return hashlib.sha256(image_data).hexdigest()
            
        except Exception as e:
            logger.warning(f"Failed to hash image, using fallback: {e}")
            return hashlib.sha256(str(time.time()).encode()).hexdigest()
    
    def get_result(
        self,
        image: Union[str, Image.Image, np.ndarray],
        question: str,
        model_id: str,
        model_version: str = "1.0"
    ) -> Optional[str]:
        """Get cached inference result."""
        try:
            image_hash = self.get_image_hash(image)
            cache_key = self.get_cache_key(image_hash, question, model_id, model_version)
            return self.cache.get(cache_key)
        except Exception as e:
            logger.error(f"Failed to get cached result: {e}")
            return None
    
    def put_result(
        self,
        image: Union[str, Image.Image, np.ndarray],
        question: str,
        result: str,
        model_id: str,
        model_version: str = "1.0",
        ttl: Optional[float] = None
    ) -> bool:
        """Cache inference result."""
        try:
            image_hash = self.get_image_hash(image)
            cache_key = self.get_cache_key(image_hash, question, model_id, model_version)
            return self.cache.put(cache_key, result, ttl)
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


class CacheManager:
    """Manages all caching systems."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        enable_model_cache: bool = True,
        enable_inference_cache: bool = True,
        enable_memory_optimization: bool = True
    ):
        """Initialize cache manager.
        
        Args:
            cache_dir: Base directory for caches
            enable_model_cache: Enable model caching
            enable_inference_cache: Enable inference result caching
            enable_memory_optimization: Enable memory optimization
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.caches = {}
        
        if enable_model_cache:
            model_cache_dir = self.cache_dir / "models" if self.cache_dir else None
            self.caches["models"] = ModelCache(
                cache_dir=str(model_cache_dir) if model_cache_dir else None
            )
        
        if enable_inference_cache:
            self.caches["inference"] = InferenceCache()
        
        self.memory_optimization = enable_memory_optimization
    
    def get_model_cache(self) -> Optional[ModelCache]:
        """Get model cache."""
        return self.caches.get("models")
    
    def get_inference_cache(self) -> Optional[InferenceCache]:
        """Get inference cache."""
        return self.caches.get("inference")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get all cache statistics."""
        stats = {}
        for name, cache in self.caches.items():
            if hasattr(cache, 'get_cache_stats'):
                stats[name] = cache.get_cache_stats()
            elif hasattr(cache, 'get_stats'):
                stats[name] = cache.get_stats()
        return stats
    
    def clear_all_caches(self):
        """Clear all caches."""
        for cache in self.caches.values():
            if hasattr(cache, 'clear'):
                cache.clear()
            elif hasattr(cache, 'memory_cache'):
                cache.memory_cache.clear()
    
    def optimize_memory(self):
        """Optimize memory usage across all caches."""
        if not self.memory_optimization:
            return
        
        stats = self.get_cache_stats()
        total_memory = sum(
            s.get("memory", {}).get("memory_usage_mb", 0)
            for s in stats.values()
        )
        
        # If using too much memory, trigger cleanup
        if total_memory > 2048:  # 2GB threshold
            logger.info(f"High memory usage detected ({total_memory:.1f}MB), optimizing caches")
            
            for cache in self.caches.values():
                if hasattr(cache, 'memory_cache'):
                    cache.memory_cache._cleanup_expired()
    
    def shutdown(self):
        """Shutdown all caches."""
        for cache in self.caches.values():
            if hasattr(cache, 'shutdown'):
                cache.shutdown()
            elif hasattr(cache, 'memory_cache') and hasattr(cache.memory_cache, 'shutdown'):
                cache.memory_cache.shutdown()


def create_cache_manager(
    cache_dir: str = "~/.fastvlm/cache",
    config: Optional[Dict[str, Any]] = None
) -> CacheManager:
    """Create cache manager with configuration.
    
    Args:
        cache_dir: Cache directory path
        config: Cache configuration
        
    Returns:
        Configured cache manager
    """
    cache_path = Path(cache_dir).expanduser()
    
    default_config = {
        "enable_model_cache": True,
        "enable_inference_cache": True,
        "enable_memory_optimization": True
    }
    
    if config:
        default_config.update(config)
    
    return CacheManager(
        cache_dir=str(cache_path),
        **default_config
    )