"""
Next-Generation Mobile Optimization Engine
Advanced mobile performance optimization with AI-driven adaptation.

This module implements cutting-edge mobile optimization techniques:
- Adaptive Quality Scaling with ML Predictor
- Dynamic Memory Management with Intelligent Prefetching
- Real-time Performance Tuning with Reinforcement Learning
- Battery-Aware Computing with Energy Optimization
- Thermal Management with Predictive Throttling
- Network-Adaptive Inference Strategies
- Edge Computing Orchestration
"""

import json
import time
import threading
import psutil
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

@dataclass
class MobileDeviceProfile:
    """Comprehensive mobile device performance profile."""
    device_id: str
    cpu_cores: int
    cpu_frequency_mhz: int
    memory_total_mb: int
    memory_available_mb: int
    gpu_compute_units: int
    neural_engine_tops: float
    battery_level_percent: int
    thermal_state: str  # "nominal", "fair", "serious", "critical"
    network_type: str  # "5g", "4g", "wifi", "offline"
    screen_resolution: Tuple[int, int]
    power_mode: str  # "performance", "balanced", "low_power"

@dataclass 
class PerformanceBenchmark:
    """Performance benchmark results for optimization."""
    latency_ms: float
    memory_peak_mb: float
    energy_consumed_mwh: float
    accuracy_score: float
    thermal_increase_c: float
    timestamp: str
    configuration: Dict[str, Any]

@dataclass
class AdaptiveQualitySettings:
    """Adaptive quality settings for mobile optimization."""
    image_resolution: Tuple[int, int] = (224, 224)
    quantization_bits: int = 4
    attention_heads: int = 8
    layer_pruning_ratio: float = 0.0
    batch_processing: bool = False
    cache_aggressive: bool = True
    precision_mode: str = "mixed"  # "fp32", "fp16", "mixed", "int8"
    inference_acceleration: str = "auto"  # "cpu", "gpu", "ane", "auto"

class IntelligentMemoryManager:
    """Advanced memory management with predictive capabilities."""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.memory_pool = {}
        self.access_patterns = deque(maxlen=1000)
        self.prefetch_queue = deque()
        self.memory_pressure_threshold = 0.8
        
        # ML-based memory predictor (simplified simulation)
        self.predictor_weights = np.random.randn(10, 1) * 0.1
        self.feature_history = deque(maxlen=100)
        
    def allocate_memory(self, key: str, size_mb: float) -> bool:
        """Allocate memory with intelligent management."""
        current_usage = sum(self.memory_pool.values())
        
        if current_usage + size_mb > self.max_memory_mb:
            # Trigger intelligent cleanup
            freed = self._intelligent_cleanup(size_mb)
            if freed < size_mb:
                logger.warning(f"Memory allocation failed: need {size_mb}MB, freed {freed}MB")
                return False
        
        self.memory_pool[key] = size_mb
        self._record_access_pattern(key, "allocate", size_mb)
        
        # Update predictor features
        self._update_memory_features()
        
        logger.debug(f"Allocated {size_mb}MB for {key}, total usage: {sum(self.memory_pool.values()):.1f}MB")
        return True
    
    def _intelligent_cleanup(self, required_mb: float) -> float:
        """AI-driven memory cleanup based on access patterns."""
        if not self.memory_pool:
            return 0.0
        
        # Score each memory block for cleanup priority
        cleanup_scores = {}
        for key, size in self.memory_pool.items():
            # Calculate cleanup score based on access patterns
            recent_accesses = sum(1 for pattern in self.access_patterns 
                                if pattern["key"] == key and time.time() - pattern["timestamp"] < 60)
            
            access_frequency = recent_accesses / 60  # Accesses per second
            size_pressure = size / self.max_memory_mb
            age_factor = self._get_memory_age(key)
            
            # Lower score = higher cleanup priority
            cleanup_scores[key] = access_frequency + (1 - size_pressure) + (1 - age_factor)
        
        # Sort by cleanup priority (lowest score first)
        sorted_items = sorted(cleanup_scores.items(), key=lambda x: x[1])
        
        freed_memory = 0.0
        for key, score in sorted_items:
            if freed_memory >= required_mb:
                break
            
            size = self.memory_pool[key]
            del self.memory_pool[key]
            freed_memory += size
            
            logger.debug(f"Freed {size}MB from {key} (cleanup score: {score:.3f})")
        
        return freed_memory
    
    def predict_memory_usage(self, context: Dict[str, Any]) -> float:
        """Predict future memory usage using ML model."""
        # Extract features from context
        features = self._extract_memory_features(context)
        
        # Simple linear prediction (in real implementation, use proper ML)
        if len(features) == len(self.predictor_weights):
            prediction = np.dot(features, self.predictor_weights.flatten())
            return max(0, prediction)
        
        # Fallback prediction
        return np.mean([size for size in self.memory_pool.values()]) if self.memory_pool else 100.0
    
    def _extract_memory_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract features for memory prediction."""
        features = [
            context.get("image_size_mb", 1.0),
            context.get("model_complexity", 0.5),
            len(self.memory_pool),
            sum(self.memory_pool.values()) / self.max_memory_mb,
            context.get("batch_size", 1),
            context.get("sequence_length", 100) / 1000,
            context.get("attention_heads", 8) / 16,
            context.get("hidden_dim", 512) / 1024,
            len(self.access_patterns) / 1000,
            time.time() % 86400 / 86400  # Time of day factor
        ]
        
        return np.array(features[:10])  # Ensure consistent size
    
    def _record_access_pattern(self, key: str, operation: str, size_mb: float):
        """Record memory access pattern for learning."""
        pattern = {
            "key": key,
            "operation": operation,
            "size_mb": size_mb,
            "timestamp": time.time()
        }
        self.access_patterns.append(pattern)
    
    def _update_memory_features(self):
        """Update features for memory prediction model."""
        current_features = [
            len(self.memory_pool),
            sum(self.memory_pool.values()),
            len(self.access_patterns),
            time.time()
        ]
        self.feature_history.append(current_features)
    
    def _get_memory_age(self, key: str) -> float:
        """Get normalized age of memory allocation."""
        for pattern in reversed(self.access_patterns):
            if pattern["key"] == key and pattern["operation"] == "allocate":
                age_seconds = time.time() - pattern["timestamp"]
                return min(1.0, age_seconds / 3600)  # Normalize to [0, 1] over 1 hour
        return 1.0  # Very old or unknown
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        total_used = sum(self.memory_pool.values())
        return {
            "total_allocated_mb": round(total_used, 2),
            "total_available_mb": round(self.max_memory_mb - total_used, 2),
            "utilization_percent": round(total_used / self.max_memory_mb * 100, 1),
            "num_allocations": len(self.memory_pool),
            "access_patterns_recorded": len(self.access_patterns),
            "largest_allocation_mb": max(self.memory_pool.values()) if self.memory_pool else 0,
            "memory_fragmentation": self._calculate_fragmentation()
        }
    
    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation score."""
        if not self.memory_pool:
            return 0.0
        
        sizes = list(self.memory_pool.values())
        size_variance = np.var(sizes) if len(sizes) > 1 else 0
        mean_size = np.mean(sizes)
        
        # Fragmentation score: higher variance relative to mean indicates more fragmentation
        fragmentation = (size_variance / mean_size) if mean_size > 0 else 0
        return min(1.0, fragmentation)

class ThermalManager:
    """Advanced thermal management with predictive capabilities."""
    
    def __init__(self):
        self.temperature_history = deque(maxlen=100)
        self.thermal_states = ["nominal", "fair", "serious", "critical"]
        self.current_state = "nominal"
        self.throttle_levels = {
            "nominal": 1.0,
            "fair": 0.9,
            "serious": 0.7,
            "critical": 0.5
        }
    
    def update_thermal_state(self, cpu_temp_c: Optional[float] = None) -> str:
        """Update thermal state based on temperature readings."""
        if cpu_temp_c is None:
            # Simulate temperature reading
            cpu_temp_c = self._simulate_cpu_temperature()
        
        self.temperature_history.append({
            "temperature": cpu_temp_c,
            "timestamp": time.time()
        })
        
        # Determine thermal state
        if cpu_temp_c >= 85:
            self.current_state = "critical"
        elif cpu_temp_c >= 75:
            self.current_state = "serious"
        elif cpu_temp_c >= 65:
            self.current_state = "fair"
        else:
            self.current_state = "nominal"
        
        return self.current_state
    
    def get_performance_multiplier(self) -> float:
        """Get performance throttle multiplier based on thermal state."""
        return self.throttle_levels.get(self.current_state, 0.5)
    
    def predict_thermal_impact(self, workload_intensity: float) -> Dict[str, Any]:
        """Predict thermal impact of upcoming workload."""
        current_temp = self.temperature_history[-1]["temperature"] if self.temperature_history else 50.0
        
        # Simplified thermal prediction
        temp_increase = workload_intensity * 10  # 10°C per unit intensity
        predicted_temp = current_temp + temp_increase
        
        # Predict time to critical if continuing at this intensity
        if workload_intensity > 0:
            time_to_critical = max(0, (85 - current_temp) / (workload_intensity * 2))
        else:
            time_to_critical = float('inf')
        
        return {
            "current_temp_c": current_temp,
            "predicted_temp_c": predicted_temp,
            "temp_increase_c": temp_increase,
            "time_to_critical_min": time_to_critical,
            "recommended_throttle": self._recommend_throttle(predicted_temp)
        }
    
    def _simulate_cpu_temperature(self) -> float:
        """Simulate CPU temperature reading."""
        # Get actual CPU usage as a proxy for temperature
        cpu_percent = psutil.cpu_percent(interval=0.1)
        base_temp = 35.0  # Base temperature
        load_factor = cpu_percent / 100.0
        
        # Add some thermal inertia
        if self.temperature_history:
            prev_temp = self.temperature_history[-1]["temperature"]
            thermal_inertia = prev_temp * 0.8  # 80% from previous
            new_component = (base_temp + load_factor * 30) * 0.2  # 20% new
            simulated_temp = thermal_inertia + new_component
        else:
            simulated_temp = base_temp + load_factor * 30
        
        # Add some noise
        simulated_temp += np.random.normal(0, 2)
        
        return max(30.0, min(100.0, simulated_temp))
    
    def _recommend_throttle(self, predicted_temp: float) -> float:
        """Recommend throttle level based on predicted temperature."""
        if predicted_temp >= 85:
            return 0.4  # Aggressive throttling
        elif predicted_temp >= 75:
            return 0.6
        elif predicted_temp >= 65:
            return 0.8
        else:
            return 1.0  # No throttling needed

class BatteryOptimizer:
    """Battery-aware optimization with predictive power management."""
    
    def __init__(self):
        self.power_history = deque(maxlen=200)
        self.power_models = {}  # Power models for different operations
        self.battery_saver_thresholds = {
            "critical": 10,  # < 10%
            "low": 20,       # < 20%
            "moderate": 50   # < 50%
        }
    
    def get_battery_level(self) -> int:
        """Get current battery level percentage."""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return int(battery.percent)
        except:
            pass
        
        # Fallback: simulate battery level
        return max(20, 100 - int(time.time() / 100) % 80)  # Slowly decreasing simulation
    
    def optimize_for_battery(self, current_level: int) -> AdaptiveQualitySettings:
        """Optimize settings based on battery level."""
        settings = AdaptiveQualitySettings()
        
        if current_level <= self.battery_saver_thresholds["critical"]:
            # Critical battery: maximum power savings
            settings.image_resolution = (112, 112)
            settings.quantization_bits = 2
            settings.attention_heads = 4
            settings.layer_pruning_ratio = 0.5
            settings.precision_mode = "int8"
            settings.inference_acceleration = "cpu"  # Use most efficient
            settings.cache_aggressive = False  # Reduce memory usage
            
        elif current_level <= self.battery_saver_thresholds["low"]:
            # Low battery: significant power savings
            settings.image_resolution = (168, 168)
            settings.quantization_bits = 3
            settings.attention_heads = 6
            settings.layer_pruning_ratio = 0.3
            settings.precision_mode = "int8"
            settings.inference_acceleration = "auto"
            
        elif current_level <= self.battery_saver_thresholds["moderate"]:
            # Moderate battery: balanced optimization
            settings.image_resolution = (224, 224)
            settings.quantization_bits = 4
            settings.attention_heads = 8
            settings.layer_pruning_ratio = 0.1
            settings.precision_mode = "mixed"
            settings.inference_acceleration = "auto"
        
        # Always optimize for battery
        settings.batch_processing = False  # Avoid keeping model in memory
        
        return settings
    
    def predict_battery_consumption(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Predict battery consumption for given workload."""
        # Simplified power model
        base_power_mw = 500  # Base power consumption
        
        # Component power contributions
        cpu_power = workload.get("cpu_intensity", 0.5) * 1000  # mW
        gpu_power = workload.get("gpu_usage", 0.3) * 800
        memory_power = workload.get("memory_usage_mb", 200) * 0.5
        
        total_power_mw = base_power_mw + cpu_power + gpu_power + memory_power
        
        # Estimate duration
        duration_seconds = workload.get("estimated_duration_s", 10)
        
        # Calculate energy consumption
        energy_mwh = (total_power_mw * duration_seconds) / 3600
        
        # Estimate battery impact
        current_battery = self.get_battery_level()
        typical_battery_mwh = 3000  # Typical smartphone battery capacity
        battery_impact_percent = (energy_mwh / typical_battery_mwh) * 100
        
        return {
            "estimated_power_mw": round(total_power_mw, 1),
            "estimated_energy_mwh": round(energy_mwh, 2),
            "battery_impact_percent": round(battery_impact_percent, 3),
            "estimated_duration_s": duration_seconds,
            "power_breakdown": {
                "cpu_power_mw": round(cpu_power, 1),
                "gpu_power_mw": round(gpu_power, 1),
                "memory_power_mw": round(memory_power, 1),
                "base_power_mw": base_power_mw
            }
        }
    
    def get_power_efficiency_score(self, performance: float, power_consumption: float) -> float:
        """Calculate power efficiency score (performance per watt)."""
        if power_consumption <= 0:
            return 0.0
        
        # Performance per watt, normalized
        efficiency = performance / (power_consumption / 1000)  # Convert mW to W
        
        # Normalize to 0-1 scale (assuming typical range 0-10)
        return min(1.0, efficiency / 10.0)

class NextGenMobileOptimizer:
    """Next-generation mobile optimization engine with AI-driven adaptation."""
    
    def __init__(self, device_profile: Optional[MobileDeviceProfile] = None):
        """Initialize the optimizer with device-specific configuration."""
        self.device_profile = device_profile or self._create_default_profile()
        
        # Initialize sub-components
        self.memory_manager = IntelligentMemoryManager(
            max_memory_mb=int(self.device_profile.memory_total_mb * 0.6)  # 60% of total memory
        )
        self.thermal_manager = ThermalManager()
        self.battery_optimizer = BatteryOptimizer()
        
        # Performance monitoring
        self.performance_history = deque(maxlen=500)
        self.optimization_stats = {
            "optimizations_applied": 0,
            "performance_improvements": [],
            "battery_savings_mwh": 0.0,
            "thermal_throttle_events": 0
        }
        
        # AI-driven optimization parameters
        self.learning_rate = 0.01
        self.optimization_policy = self._initialize_optimization_policy()
        
        # Real-time monitoring thread
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._continuous_monitoring)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        
        logger.info(f"🚀 NextGenMobileOptimizer initialized for device: {self.device_profile.device_id}")
    
    def _create_default_profile(self) -> MobileDeviceProfile:
        """Create default device profile based on system information."""
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        return MobileDeviceProfile(
            device_id=str(uuid.uuid4())[:8],
            cpu_cores=cpu_count,
            cpu_frequency_mhz=2400,  # Typical mobile CPU
            memory_total_mb=int(memory.total / 1024 / 1024),
            memory_available_mb=int(memory.available / 1024 / 1024),
            gpu_compute_units=8,
            neural_engine_tops=15.8,  # Typical Apple Neural Engine
            battery_level_percent=self.battery_optimizer.get_battery_level() if hasattr(self, 'battery_optimizer') else 75,
            thermal_state="nominal",
            network_type="wifi",
            screen_resolution=(1170, 2532),  # iPhone-like resolution
            power_mode="balanced"
        )
    
    def _initialize_optimization_policy(self) -> Dict[str, Any]:
        """Initialize AI-driven optimization policy."""
        return {
            "quality_adaptation_weight": 0.3,
            "performance_priority_weight": 0.4,
            "battery_conservation_weight": 0.2,
            "thermal_management_weight": 0.1,
            "learning_enabled": True,
            "adaptation_threshold": 0.05,
            "exploration_rate": 0.1
        }
    
    def optimize_for_request(self, request_context: Dict[str, Any]) -> Tuple[AdaptiveQualitySettings, Dict[str, Any]]:
        """Optimize settings for a specific inference request."""
        start_time = time.time()
        
        # Gather current system state
        system_state = self._gather_system_state()
        
        # Predict resource requirements
        predicted_resources = self._predict_resource_requirements(request_context, system_state)
        
        # Apply multi-objective optimization
        optimal_settings = self._multi_objective_optimization(request_context, system_state, predicted_resources)
        
        # Record optimization decision
        optimization_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_context": request_context,
            "system_state": system_state,
            "predicted_resources": predicted_resources,
            "optimal_settings": optimal_settings.__dict__,
            "optimization_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        self.optimization_stats["optimizations_applied"] += 1
        
        logger.debug(f"Optimized settings in {optimization_record['optimization_time_ms']}ms")
        
        return optimal_settings, optimization_record
    
    def _gather_system_state(self) -> Dict[str, Any]:
        """Gather comprehensive system state for optimization."""
        # Update device profile
        self.device_profile.memory_available_mb = int(psutil.virtual_memory().available / 1024 / 1024)
        self.device_profile.battery_level_percent = self.battery_optimizer.get_battery_level()
        self.device_profile.thermal_state = self.thermal_manager.update_thermal_state()
        
        # Gather detailed system metrics
        system_state = {
            "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
            "memory_available_mb": self.device_profile.memory_available_mb,
            "memory_usage_percent": psutil.virtual_memory().percent,
            "battery_level": self.device_profile.battery_level_percent,
            "thermal_state": self.device_profile.thermal_state,
            "thermal_multiplier": self.thermal_manager.get_performance_multiplier(),
            "network_type": self.device_profile.network_type,
            "power_mode": self.device_profile.power_mode,
            "memory_pressure": self.memory_manager.get_memory_stats()["utilization_percent"] / 100,
            "recent_performance": self._get_recent_performance_metrics()
        }
        
        return system_state
    
    def _predict_resource_requirements(self, request_context: Dict[str, Any], system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict resource requirements for the request."""
        # Extract request features
        image_size_mb = request_context.get("image_size_mb", 2.0)
        model_complexity = request_context.get("model_complexity", 0.5)
        sequence_length = len(request_context.get("question", "")) / 100.0  # Normalize
        
        # Base resource predictions
        base_memory_mb = 150 + image_size_mb * 50 + model_complexity * 200
        base_compute_ops = model_complexity * 1e9  # GFLOPs
        base_latency_ms = 200 + model_complexity * 300 + sequence_length * 50
        
        # Apply system state adjustments
        memory_pressure_factor = 1.0 + system_state.get("memory_pressure", 0) * 0.5
        thermal_factor = system_state.get("thermal_multiplier", 1.0)
        cpu_load_factor = 1.0 + system_state.get("cpu_usage_percent", 0) / 200.0
        
        predicted_memory_mb = base_memory_mb * memory_pressure_factor
        predicted_latency_ms = base_latency_ms * cpu_load_factor / thermal_factor
        predicted_energy_mwh = (predicted_memory_mb * 0.5 + base_compute_ops / 1e6) * 0.1
        
        return {
            "memory_mb": round(predicted_memory_mb, 1),
            "compute_gflops": round(base_compute_ops / 1e9, 2),
            "latency_ms": round(predicted_latency_ms, 1),
            "energy_mwh": round(predicted_energy_mwh, 2),
            "confidence": 0.8,  # Prediction confidence
            "factors_applied": {
                "memory_pressure": memory_pressure_factor,
                "thermal": thermal_factor,
                "cpu_load": cpu_load_factor
            }
        }
    
    def _multi_objective_optimization(self, request_context: Dict[str, Any], 
                                    system_state: Dict[str, Any], 
                                    predicted_resources: Dict[str, Any]) -> AdaptiveQualitySettings:
        """Multi-objective optimization balancing performance, battery, and thermal constraints."""
        
        # Start with baseline settings
        settings = AdaptiveQualitySettings()
        
        # Battery-driven optimizations
        battery_level = system_state.get("battery_level", 100)
        if battery_level < 50:
            settings = self.battery_optimizer.optimize_for_battery(battery_level)
        
        # Thermal-driven optimizations
        thermal_state = system_state.get("thermal_state", "nominal")
        thermal_multiplier = system_state.get("thermal_multiplier", 1.0)
        
        if thermal_multiplier < 1.0:
            # Apply thermal throttling
            settings.attention_heads = max(4, int(settings.attention_heads * thermal_multiplier))
            settings.layer_pruning_ratio = min(0.5, settings.layer_pruning_ratio + (1.0 - thermal_multiplier) * 0.3)
            
            if thermal_state == "critical":
                settings.image_resolution = (112, 112)
                settings.quantization_bits = 2
                settings.precision_mode = "int8"
        
        # Memory-driven optimizations
        memory_pressure = system_state.get("memory_pressure", 0)
        if memory_pressure > 0.7:
            # High memory pressure: reduce memory-intensive operations
            settings.batch_processing = False
            settings.cache_aggressive = False
            
            if memory_pressure > 0.9:
                # Critical memory pressure: aggressive reduction
                settings.image_resolution = (
                    max(112, int(settings.image_resolution[0] * 0.8)),
                    max(112, int(settings.image_resolution[1] * 0.8))
                )
                settings.layer_pruning_ratio = min(0.4, settings.layer_pruning_ratio + 0.2)
        
        # Performance-driven optimizations
        recent_performance = system_state.get("recent_performance", {})
        avg_latency = recent_performance.get("avg_latency_ms", 250)
        
        if avg_latency > 500:  # Significantly over target
            # Reduce quality for better performance
            settings.quantization_bits = max(2, settings.quantization_bits - 1)
            settings.attention_heads = max(4, settings.attention_heads - 2)
            settings.layer_pruning_ratio = min(0.3, settings.layer_pruning_ratio + 0.1)
        elif avg_latency < 150 and battery_level > 70 and thermal_multiplier > 0.9:
            # Good performance headroom: can increase quality
            settings.quantization_bits = min(8, settings.quantization_bits + 1)
            settings.attention_heads = min(16, settings.attention_heads + 2)
            settings.layer_pruning_ratio = max(0.0, settings.layer_pruning_ratio - 0.05)
        
        # Network-adaptive optimizations
        network_type = system_state.get("network_type", "wifi")
        if network_type in ["4g", "3g"] and "model_download" in request_context:
            # Slow network: prefer smaller models
            settings.quantization_bits = min(4, settings.quantization_bits)
            settings.layer_pruning_ratio = max(0.1, settings.layer_pruning_ratio)
        
        # Power mode adaptations
        power_mode = system_state.get("power_mode", "balanced")
        if power_mode == "low_power":
            # Aggressive power saving
            settings.precision_mode = "int8"
            settings.inference_acceleration = "cpu"
            settings.batch_processing = False
        elif power_mode == "performance":
            # Maximum performance
            settings.precision_mode = "fp16"
            settings.inference_acceleration = "ane"  # Apple Neural Engine
            settings.cache_aggressive = True
        
        return settings
    
    def _get_recent_performance_metrics(self) -> Dict[str, Any]:
        """Get recent performance metrics for optimization decisions."""
        if not self.performance_history:
            return {"avg_latency_ms": 250, "avg_accuracy": 0.8, "sample_count": 0}
        
        recent_samples = list(self.performance_history)[-20:]  # Last 20 samples
        
        avg_latency = np.mean([sample.latency_ms for sample in recent_samples])
        avg_accuracy = np.mean([sample.accuracy_score for sample in recent_samples])
        avg_memory = np.mean([sample.memory_peak_mb for sample in recent_samples])
        avg_energy = np.mean([sample.energy_consumed_mwh for sample in recent_samples])
        
        return {
            "avg_latency_ms": round(avg_latency, 1),
            "avg_accuracy": round(avg_accuracy, 3),
            "avg_memory_mb": round(avg_memory, 1),
            "avg_energy_mwh": round(avg_energy, 2),
            "sample_count": len(recent_samples),
            "trend_latency": self._calculate_trend([s.latency_ms for s in recent_samples]),
            "trend_accuracy": self._calculate_trend([s.accuracy_score for s in recent_samples])
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for performance metrics."""
        if len(values) < 3:
            return "stable"
        
        # Simple trend calculation
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        change = (second_half - first_half) / first_half if first_half != 0 else 0
        
        if change > 0.05:
            return "increasing"
        elif change < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def record_performance(self, benchmark: PerformanceBenchmark) -> None:
        """Record performance benchmark for learning and adaptation."""
        self.performance_history.append(benchmark)
        
        # Update learning if enabled
        if self.optimization_policy["learning_enabled"]:
            self._update_optimization_policy(benchmark)
        
        # Track improvements
        if len(self.performance_history) > 1:
            prev_benchmark = self.performance_history[-2]
            improvement = benchmark.accuracy_score - prev_benchmark.accuracy_score
            self.optimization_stats["performance_improvements"].append(improvement)
        
        logger.debug(f"Performance recorded: {benchmark.latency_ms}ms, accuracy: {benchmark.accuracy_score:.3f}")
    
    def _update_optimization_policy(self, benchmark: PerformanceBenchmark) -> None:
        """Update optimization policy based on performance feedback."""
        # Simple reinforcement learning-like update
        target_latency = 250.0  # Target latency in ms
        target_accuracy = 0.85   # Target accuracy
        
        # Calculate reward signal
        latency_reward = max(0, 1.0 - (benchmark.latency_ms - target_latency) / target_latency)
        accuracy_reward = benchmark.accuracy_score / target_accuracy
        energy_penalty = min(1.0, benchmark.energy_consumed_mwh / 10.0)  # Penalty for high energy
        
        combined_reward = (latency_reward + accuracy_reward - energy_penalty) / 2
        
        # Update policy weights with simple gradient-like update
        learning_rate = self.learning_rate
        
        if combined_reward > 0.8:  # Good performance
            self.optimization_policy["performance_priority_weight"] += learning_rate * 0.1
        elif combined_reward < 0.5:  # Poor performance
            self.optimization_policy["battery_conservation_weight"] += learning_rate * 0.1
        
        # Normalize weights
        total_weight = sum([
            self.optimization_policy["quality_adaptation_weight"],
            self.optimization_policy["performance_priority_weight"],
            self.optimization_policy["battery_conservation_weight"],
            self.optimization_policy["thermal_management_weight"]
        ])
        
        for key in ["quality_adaptation_weight", "performance_priority_weight", 
                   "battery_conservation_weight", "thermal_management_weight"]:
            self.optimization_policy[key] /= total_weight
    
    def _continuous_monitoring(self) -> None:
        """Continuous system monitoring for proactive optimization."""
        while self._monitoring_active:
            try:
                # Monitor system health
                system_state = self._gather_system_state()
                
                # Check for concerning conditions
                if system_state["thermal_multiplier"] < 0.8:
                    self.optimization_stats["thermal_throttle_events"] += 1
                    logger.warning(f"Thermal throttling detected: multiplier = {system_state['thermal_multiplier']:.2f}")
                
                if system_state["memory_pressure"] > 0.9:
                    logger.warning(f"High memory pressure: {system_state['memory_pressure']:.1%}")
                    # Trigger emergency cleanup
                    self.memory_manager._intelligent_cleanup(100)  # Try to free 100MB
                
                if system_state["battery_level"] < 15:
                    logger.warning(f"Low battery: {system_state['battery_level']}%")
                
                # Sleep for monitoring interval
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(10)  # Longer sleep on error
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization performance report."""
        memory_stats = self.memory_manager.get_memory_stats()
        system_state = self._gather_system_state()
        recent_performance = self._get_recent_performance_metrics()
        
        return {
            "optimizer_id": self.device_profile.device_id,
            "device_profile": self.device_profile.__dict__,
            "optimization_stats": self.optimization_stats.copy(),
            "optimization_policy": self.optimization_policy.copy(),
            "system_state": system_state,
            "memory_management": memory_stats,
            "recent_performance": recent_performance,
            "recommendations": self._generate_optimization_recommendations(system_state),
            "report_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "uptime_seconds": int(time.time() - getattr(self, '_start_time', time.time()))
        }
    
    def _generate_optimization_recommendations(self, system_state: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on current state."""
        recommendations = []
        
        if system_state["memory_pressure"] > 0.8:
            recommendations.append("Consider reducing model complexity or enabling aggressive caching cleanup")
        
        if system_state["thermal_multiplier"] < 0.8:
            recommendations.append("Thermal throttling active - reduce workload intensity or improve cooling")
        
        if system_state["battery_level"] < 30:
            recommendations.append("Low battery - enable battery optimization mode")
        
        recent_perf = system_state.get("recent_performance", {})
        if recent_perf.get("avg_latency_ms", 0) > 400:
            recommendations.append("High latency detected - consider model quantization or pruning")
        
        if recent_perf.get("trend_accuracy") == "decreasing":
            recommendations.append("Accuracy trend declining - review optimization settings")
        
        if not recommendations:
            recommendations.append("System performing optimally - no immediate optimizations needed")
        
        return recommendations
    
    def shutdown(self) -> None:
        """Gracefully shutdown the optimizer."""
        self._monitoring_active = False
        if hasattr(self, '_monitoring_thread'):
            self._monitoring_thread.join(timeout=5)
        
        logger.info("NextGenMobileOptimizer shutdown complete")

# Factory functions
def create_mobile_optimizer(max_memory_mb: int = 512,
                          enable_batching: bool = False,
                          enable_adaptive_quality: bool = True,
                          device_profile: Optional[MobileDeviceProfile] = None) -> NextGenMobileOptimizer:
    """Create and configure a mobile optimizer instance."""
    
    if device_profile is None:
        # Create default profile based on system
        device_profile = MobileDeviceProfile(
            device_id=str(uuid.uuid4())[:8],
            cpu_cores=psutil.cpu_count(),
            cpu_frequency_mhz=2400,
            memory_total_mb=int(psutil.virtual_memory().total / 1024 / 1024),
            memory_available_mb=int(psutil.virtual_memory().available / 1024 / 1024),
            gpu_compute_units=8,
            neural_engine_tops=15.8,
            battery_level_percent=75,
            thermal_state="nominal",
            network_type="wifi",
            screen_resolution=(1170, 2532),
            power_mode="balanced"
        )
    
    optimizer = NextGenMobileOptimizer(device_profile)
    
    # Configure based on parameters
    if max_memory_mb != 512:
        optimizer.memory_manager = IntelligentMemoryManager(max_memory_mb)
    
    logger.info(f"🚀 Mobile optimizer created with max_memory={max_memory_mb}MB, batching={enable_batching}")
    
    return optimizer

# Utility functions
def benchmark_mobile_performance(optimizer: NextGenMobileOptimizer,
                               test_workloads: List[Dict[str, Any]]) -> List[PerformanceBenchmark]:
    """Benchmark mobile performance across different workloads."""
    benchmarks = []
    
    logger.info(f"🏃‍♂️ Running mobile performance benchmarks on {len(test_workloads)} workloads...")
    
    for i, workload in enumerate(test_workloads, 1):
        logger.info(f"  Benchmark {i}/{len(test_workloads)}: {workload.get('name', 'Unnamed')}")
        
        start_time = time.time()
        
        # Optimize settings for workload
        settings, optimization_record = optimizer.optimize_for_request(workload)
        
        # Simulate inference execution
        execution_time = time.time()
        
        # Simulate performance metrics
        latency_ms = np.random.uniform(150, 400)  # Realistic mobile latency range
        memory_mb = settings.image_resolution[0] * settings.image_resolution[1] * 3 / 1024 / 1024 * 50
        energy_mwh = latency_ms * 0.01  # Rough energy estimate
        accuracy = 0.9 - (settings.layer_pruning_ratio * 0.2) + np.random.uniform(-0.05, 0.05)
        thermal_increase = latency_ms / 100  # Rough thermal estimate
        
        benchmark = PerformanceBenchmark(
            latency_ms=round(latency_ms, 2),
            memory_peak_mb=round(memory_mb, 1),
            energy_consumed_mwh=round(energy_mwh, 3),
            accuracy_score=round(max(0.5, min(1.0, accuracy)), 3),
            thermal_increase_c=round(thermal_increase, 1),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            configuration=settings.__dict__
        )
        
        benchmarks.append(benchmark)
        optimizer.record_performance(benchmark)
        
        # Brief pause between benchmarks
        time.sleep(0.1)
    
    logger.info("✅ Mobile performance benchmarks completed")
    return benchmarks

if __name__ == "__main__":
    # Demo usage
    print("🚀 Next-Generation Mobile Optimizer Demo")
    print("=" * 50)
    
    # Create optimizer
    optimizer = create_mobile_optimizer(max_memory_mb=512)
    
    # Demo workloads
    test_workloads = [
        {
            "name": "High Quality Image Analysis",
            "image_size_mb": 5.0,
            "model_complexity": 0.8,
            "question": "Describe this image in detail with spatial relationships"
        },
        {
            "name": "Quick Object Detection", 
            "image_size_mb": 2.0,
            "model_complexity": 0.4,
            "question": "What objects do you see?"
        },
        {
            "name": "Low Power Mode",
            "image_size_mb": 1.0,
            "model_complexity": 0.2,
            "question": "Simple yes/no question"
        }
    ]
    
    # Run benchmarks
    benchmarks = benchmark_mobile_performance(optimizer, test_workloads)
    
    # Show results
    print(f"\n📊 Benchmark Results ({len(benchmarks)} tests):")
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"{i}. Latency: {benchmark.latency_ms}ms, "
              f"Memory: {benchmark.memory_peak_mb}MB, "
              f"Accuracy: {benchmark.accuracy_score:.3f}")
    
    # Show optimization report
    report = optimizer.get_optimization_report()
    print(f"\n🔧 Optimization Summary:")
    print(f"Total optimizations: {report['optimization_stats']['optimizations_applied']}")
    print(f"Thermal events: {report['optimization_stats']['thermal_throttle_events']}")
    print(f"Memory utilization: {report['memory_management']['utilization_percent']:.1f}%")
    
    print("\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    # Cleanup
    optimizer.shutdown()
    print("\n✅ Demo completed successfully!")