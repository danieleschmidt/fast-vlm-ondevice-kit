"""
Mobile optimization engine for FastVLM on-device deployment.

Implements advanced optimizations for mobile devices including
memory management, battery optimization, and adaptive performance.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import weakref

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels for mobile deployment."""
    BATTERY_SAVER = "battery_saver"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    ADAPTIVE = "adaptive"


class DeviceProfile(Enum):
    """Device performance profiles."""
    HIGH_END = "high_end"      # iPhone 15 Pro, iPad Pro M2
    MID_RANGE = "mid_range"    # iPhone 14, iPad Air
    LOW_END = "low_end"        # iPhone 12, older devices
    AUTO_DETECT = "auto_detect"


@dataclass
class MobileOptimizationConfig:
    """Configuration for mobile optimization."""
    device_profile: DeviceProfile = DeviceProfile.AUTO_DETECT
    optimization_level: OptimizationLevel = OptimizationLevel.ADAPTIVE
    
    # Performance targets
    target_latency_ms: float = 250.0
    max_memory_mb: float = 1500.0
    target_fps: float = 4.0  # For video processing
    
    # Battery optimization
    enable_battery_optimization: bool = True
    thermal_throttling: bool = True
    adaptive_quality: bool = True
    
    # Memory management
    enable_memory_pool: bool = True
    aggressive_gc: bool = True
    memory_pressure_threshold: float = 0.8
    
    # Caching and prefetching
    enable_smart_caching: bool = True
    prefetch_aggressive: bool = False
    cache_eviction_policy: str = "lru"
    
    # Neural Engine optimization
    use_neural_engine: bool = True
    fallback_to_gpu: bool = True
    compute_unit_allocation: Dict[str, str] = field(default_factory=lambda: {
        "vision_encoder": "neural_engine",
        "text_encoder": "cpu",
        "fusion": "gpu", 
        "decoder": "neural_engine"
    })


@dataclass
class DeviceCapabilities:
    """Device hardware capabilities."""
    chip_name: str = "Unknown"
    neural_engine_cores: int = 0
    gpu_cores: int = 0
    cpu_cores: int = 0
    memory_gb: float = 0.0
    storage_gb: float = 0.0
    thermal_state: str = "nominal"
    battery_level: float = 1.0
    low_power_mode: bool = False


class MobileResourceManager:
    """Manages mobile device resources and optimization."""
    
    def __init__(self, config: MobileOptimizationConfig):
        self.config = config
        self.device_caps = DeviceCapabilities()
        self.resource_locks = {}
        self.memory_pool = {}
        self.performance_history = []
        self.adaptive_params = {}
        
        # Initialize monitoring
        self._setup_resource_monitoring()
        self._detect_device_capabilities()
        
    def _setup_resource_monitoring(self):
        """Set up resource monitoring."""
        logger.info("Setting up mobile resource monitoring")
        
        # Memory monitoring
        self.memory_tracker = {
            "peak_usage": 0.0,
            "current_usage": 0.0,
            "pressure_events": 0,
            "gc_count": 0
        }
        
        # Performance monitoring
        self.perf_tracker = {
            "inference_times": [],
            "thermal_events": 0,
            "throttle_events": 0,
            "battery_drain_rate": 0.0
        }
        
    def _detect_device_capabilities(self):
        """Detect device capabilities."""
        logger.info("Detecting device capabilities")
        
        if self.config.device_profile == DeviceProfile.AUTO_DETECT:
            # Simulate device detection
            self.device_caps = DeviceCapabilities(
                chip_name="A17 Pro",  # Simulated
                neural_engine_cores=16,
                gpu_cores=6,
                cpu_cores=6,
                memory_gb=8.0,
                storage_gb=256.0,
                thermal_state="nominal",
                battery_level=0.85,
                low_power_mode=False
            )
        else:
            # Use predefined profile
            self._load_device_profile()
            
    def _load_device_profile(self):
        """Load predefined device profile."""
        profiles = {
            DeviceProfile.HIGH_END: DeviceCapabilities(
                chip_name="A17 Pro", neural_engine_cores=16, gpu_cores=6,
                cpu_cores=6, memory_gb=8.0, storage_gb=256.0
            ),
            DeviceProfile.MID_RANGE: DeviceCapabilities(
                chip_name="A16", neural_engine_cores=16, gpu_cores=5,
                cpu_cores=6, memory_gb=6.0, storage_gb=128.0
            ),
            DeviceProfile.LOW_END: DeviceCapabilities(
                chip_name="A14", neural_engine_cores=16, gpu_cores=4,
                cpu_cores=6, memory_gb=4.0, storage_gb=64.0
            )
        }
        
        self.device_caps = profiles.get(self.config.device_profile, DeviceCapabilities())
        
    def optimize_for_device(self) -> Dict[str, Any]:
        """Optimize configuration for current device."""
        logger.info(f"Optimizing for device: {self.device_caps.chip_name}")
        
        optimizations = {
            "compute_allocation": self._optimize_compute_allocation(),
            "memory_settings": self._optimize_memory_settings(),
            "quality_settings": self._optimize_quality_settings(),
            "batch_settings": self._optimize_batch_settings()
        }
        
        logger.info(f"Applied {len(optimizations)} optimization categories")
        return optimizations
        
    def _optimize_compute_allocation(self) -> Dict[str, str]:
        """Optimize compute unit allocation."""
        allocation = self.config.compute_unit_allocation.copy()
        
        # Adjust based on device capabilities
        if self.device_caps.neural_engine_cores == 0:
            # Fallback if no Neural Engine
            allocation["vision_encoder"] = "gpu"
            allocation["decoder"] = "gpu"
            
        if self.device_caps.gpu_cores < 4:
            # Use CPU for fusion on low-end devices
            allocation["fusion"] = "cpu"
            
        # Adaptive optimization based on thermal state
        if self.device_caps.thermal_state in ["warm", "hot"]:
            allocation["text_encoder"] = "cpu"  # Reduce GPU load
            
        return allocation
        
    def _optimize_memory_settings(self) -> Dict[str, Any]:
        """Optimize memory usage settings."""
        memory_factor = self.device_caps.memory_gb / 8.0  # Normalize to 8GB
        
        return {
            "max_memory_mb": min(self.config.max_memory_mb, 
                               self.device_caps.memory_gb * 1024 * 0.7),  # 70% of total
            "cache_size_mb": max(50, int(100 * memory_factor)),
            "batch_size": 1 if memory_factor < 0.5 else min(4, int(2 * memory_factor)),
            "aggressive_gc": memory_factor < 0.75,
            "memory_pool_size": int(200 * memory_factor)
        }
        
    def _optimize_quality_settings(self) -> Dict[str, Any]:
        """Optimize quality settings based on device and optimization level."""
        settings = {}
        
        if self.config.optimization_level == OptimizationLevel.BATTERY_SAVER:
            settings.update({
                "image_size": (224, 224),
                "max_tokens": 50,
                "precision": "int4",
                "enable_pruning": True
            })
        elif self.config.optimization_level == OptimizationLevel.PERFORMANCE:
            settings.update({
                "image_size": (448, 448) if self.device_caps.memory_gb > 6 else (336, 336),
                "max_tokens": 150,
                "precision": "fp16",
                "enable_pruning": False
            })
        else:  # Balanced or Adaptive
            settings.update({
                "image_size": (336, 336),
                "max_tokens": 100,
                "precision": "int8",
                "enable_pruning": True
            })
            
        return settings
        
    def _optimize_batch_settings(self) -> Dict[str, int]:
        """Optimize batching settings."""
        if self.device_caps.memory_gb < 4:
            return {"batch_size": 1, "prefetch_size": 1}
        elif self.device_caps.memory_gb < 6:
            return {"batch_size": 2, "prefetch_size": 2}
        else:
            return {"batch_size": 4, "prefetch_size": 3}


class AdaptivePerformanceController:
    """Controls adaptive performance based on runtime conditions."""
    
    def __init__(self, resource_manager: MobileResourceManager):
        self.resource_manager = resource_manager
        self.adaptation_history = []
        self.current_mode = OptimizationLevel.BALANCED
        self.monitoring_active = False
        self._lock = threading.Lock()
        
    def start_monitoring(self):
        """Start adaptive performance monitoring."""
        if self.monitoring_active:
            return
            
        logger.info("Starting adaptive performance monitoring")
        self.monitoring_active = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_performance, 
            daemon=True
        )
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop adaptive performance monitoring."""
        logger.info("Stopping adaptive performance monitoring")
        self.monitoring_active = False
        
    def _monitor_performance(self):
        """Monitor performance and adapt settings."""
        while self.monitoring_active:
            try:
                self._check_adaptation_triggers()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                
    def _check_adaptation_triggers(self):
        """Check if adaptation is needed."""
        with self._lock:
            current_state = self._get_current_state()
            adaptation_needed = self._should_adapt(current_state)
            
            if adaptation_needed:
                new_mode = self._select_optimization_mode(current_state)
                if new_mode != self.current_mode:
                    logger.info(f"Adapting from {self.current_mode.value} to {new_mode.value}")
                    self._apply_adaptation(new_mode)
                    self.current_mode = new_mode
                    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current system state."""
        return {
            "battery_level": self.resource_manager.device_caps.battery_level,
            "thermal_state": self.resource_manager.device_caps.thermal_state,
            "memory_pressure": self.resource_manager.memory_tracker["current_usage"] / 
                             self.resource_manager.config.max_memory_mb,
            "average_latency": self._get_average_latency(),
            "low_power_mode": self.resource_manager.device_caps.low_power_mode
        }
        
    def _should_adapt(self, state: Dict[str, Any]) -> bool:
        """Determine if adaptation is needed."""
        triggers = [
            state["battery_level"] < 0.2,  # Low battery
            state["thermal_state"] in ["warm", "hot"],  # Thermal pressure
            state["memory_pressure"] > 0.8,  # Memory pressure
            state["average_latency"] > self.resource_manager.config.target_latency_ms * 1.5,
            state["low_power_mode"]  # System low power mode
        ]
        
        return any(triggers)
        
    def _select_optimization_mode(self, state: Dict[str, Any]) -> OptimizationLevel:
        """Select optimal mode based on current state."""
        if state["low_power_mode"] or state["battery_level"] < 0.15:
            return OptimizationLevel.BATTERY_SAVER
        elif state["thermal_state"] == "hot" or state["memory_pressure"] > 0.9:
            return OptimizationLevel.BATTERY_SAVER
        elif (state["battery_level"] > 0.7 and 
              state["thermal_state"] == "nominal" and 
              state["memory_pressure"] < 0.6):
            return OptimizationLevel.PERFORMANCE
        else:
            return OptimizationLevel.BALANCED
            
    def _apply_adaptation(self, new_mode: OptimizationLevel):
        """Apply adaptation to new optimization mode."""
        adaptation = {
            "timestamp": time.time(),
            "from_mode": self.current_mode.value,
            "to_mode": new_mode.value,
            "trigger_reason": self._get_adaptation_reason()
        }
        
        self.adaptation_history.append(adaptation)
        logger.info(f"Applied adaptation: {adaptation}")
        
    def _get_adaptation_reason(self) -> str:
        """Get reason for adaptation."""
        state = self._get_current_state()
        
        if state["low_power_mode"]:
            return "low_power_mode"
        elif state["battery_level"] < 0.2:
            return "low_battery"
        elif state["thermal_state"] in ["warm", "hot"]:
            return "thermal_pressure"
        elif state["memory_pressure"] > 0.8:
            return "memory_pressure"
        else:
            return "performance_optimization"
            
    def _get_average_latency(self) -> float:
        """Get average latency from recent inferences."""
        recent_times = self.resource_manager.perf_tracker["inference_times"][-10:]
        return sum(recent_times) / len(recent_times) if recent_times else 250.0


class MobileBatteryOptimizer:
    """Optimizes for battery life on mobile devices."""
    
    def __init__(self, resource_manager: MobileResourceManager):
        self.resource_manager = resource_manager
        self.power_profiles = self._load_power_profiles()
        self.current_profile = "balanced"
        
    def _load_power_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load power optimization profiles."""
        return {
            "maximum_battery": {
                "cpu_frequency": "low",
                "gpu_frequency": "low",
                "neural_engine_usage": "minimal",
                "cache_aggressive": True,
                "prefetch_disabled": True,
                "quality_reduced": True
            },
            "balanced": {
                "cpu_frequency": "auto",
                "gpu_frequency": "auto",
                "neural_engine_usage": "optimal",
                "cache_aggressive": False,
                "prefetch_disabled": False,
                "quality_reduced": False
            },
            "performance": {
                "cpu_frequency": "high",
                "gpu_frequency": "high",
                "neural_engine_usage": "maximum",
                "cache_aggressive": False,
                "prefetch_disabled": False,
                "quality_reduced": False
            }
        }
        
    def optimize_for_battery_level(self, battery_level: float) -> Dict[str, Any]:
        """Optimize settings based on battery level."""
        if battery_level < 0.15:
            profile = "maximum_battery"
        elif battery_level < 0.30:
            profile = "balanced"
        else:
            profile = "performance"
            
        if profile != self.current_profile:
            logger.info(f"Switching to battery profile: {profile}")
            self.current_profile = profile
            
        return self.power_profiles[profile]
        
    def estimate_battery_usage(self, inference_count: int, avg_latency_ms: float) -> Dict[str, float]:
        """Estimate battery usage for given inference load."""
        # Simplified battery estimation
        base_power_mw = 500  # Base power consumption
        compute_power_mw = avg_latency_ms * 2  # Power per inference
        
        total_power_mwh = (base_power_mw + compute_power_mw * inference_count) / 1000
        
        return {
            "estimated_mwh": total_power_mwh,
            "battery_percentage": total_power_mwh / 50000,  # Assuming 50Wh battery
            "inferences_per_percent": inference_count / max(0.1, total_power_mwh / 500)
        }


class MobileOptimizer:
    """Main mobile optimization coordinator."""
    
    def __init__(self, config: MobileOptimizationConfig = None):
        self.config = config or MobileOptimizationConfig()
        self.resource_manager = MobileResourceManager(self.config)
        self.performance_controller = AdaptivePerformanceController(self.resource_manager)
        self.battery_optimizer = MobileBatteryOptimizer(self.resource_manager)
        
        # Initialize optimization
        self._initialize_optimization()
        
    def _initialize_optimization(self):
        """Initialize mobile optimization."""
        logger.info("Initializing mobile optimization")
        
        # Apply device-specific optimizations
        device_opts = self.resource_manager.optimize_for_device()
        logger.info(f"Applied device optimizations: {list(device_opts.keys())}")
        
        # Start adaptive performance monitoring
        if self.config.optimization_level == OptimizationLevel.ADAPTIVE:
            self.performance_controller.start_monitoring()
            
    def get_optimized_config(self) -> Dict[str, Any]:
        """Get optimized configuration for current conditions."""
        return {
            "device_optimizations": self.resource_manager.optimize_for_device(),
            "battery_optimizations": self.battery_optimizer.optimize_for_battery_level(
                self.resource_manager.device_caps.battery_level
            ),
            "adaptive_settings": {
                "current_mode": self.performance_controller.current_mode.value,
                "monitoring_active": self.performance_controller.monitoring_active
            },
            "device_capabilities": {
                "chip": self.resource_manager.device_caps.chip_name,
                "memory_gb": self.resource_manager.device_caps.memory_gb,
                "neural_engine_cores": self.resource_manager.device_caps.neural_engine_cores
            }
        }
        
    def update_runtime_conditions(self, battery_level: float = None, thermal_state: str = None):
        """Update runtime conditions for adaptive optimization."""
        if battery_level is not None:
            self.resource_manager.device_caps.battery_level = battery_level
            
        if thermal_state is not None:
            self.resource_manager.device_caps.thermal_state = thermal_state
            
        logger.debug(f"Updated conditions - Battery: {battery_level}, Thermal: {thermal_state}")
        
    def record_inference_metrics(self, latency_ms: float, memory_mb: float):
        """Record inference metrics for adaptive optimization."""
        self.resource_manager.perf_tracker["inference_times"].append(latency_ms)
        self.resource_manager.memory_tracker["current_usage"] = memory_mb
        
        # Keep only recent metrics
        if len(self.resource_manager.perf_tracker["inference_times"]) > 100:
            self.resource_manager.perf_tracker["inference_times"] = \
                self.resource_manager.perf_tracker["inference_times"][-50:]
                
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "optimization_config": {
                "level": self.config.optimization_level.value,
                "device_profile": self.config.device_profile.value,
                "target_latency_ms": self.config.target_latency_ms
            },
            "device_status": {
                "battery_level": self.resource_manager.device_caps.battery_level,
                "thermal_state": self.resource_manager.device_caps.thermal_state,
                "memory_gb": self.resource_manager.device_caps.memory_gb,
                "low_power_mode": self.resource_manager.device_caps.low_power_mode
            },
            "performance_metrics": {
                "average_latency_ms": self.performance_controller._get_average_latency(),
                "memory_usage_mb": self.resource_manager.memory_tracker["current_usage"],
                "adaptation_count": len(self.performance_controller.adaptation_history)
            },
            "optimizations_active": self.get_optimized_config()
        }
        
    def cleanup(self):
        """Cleanup optimization resources."""
        logger.info("Cleaning up mobile optimizer")
        self.performance_controller.stop_monitoring()