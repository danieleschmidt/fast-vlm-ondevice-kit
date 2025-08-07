#!/usr/bin/env python3
"""
Neuromorphic FastVLM Demo

Demonstrates spike-based neural processing for ultra-low power inference.
"""

import numpy as np
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fast_vlm_ondevice.neuromorphic import (
    NeuromorphicFastVLM, 
    SpikeConfig, 
    create_neuromorphic_config,
    benchmark_neuromorphic_performance
)


def main():
    """Run neuromorphic FastVLM demonstration."""
    
    print("🧠 FastVLM Neuromorphic Computing Demo")
    print("=" * 50)
    
    # Create different neuromorphic configurations
    print("\n1. Creating neuromorphic configurations...")
    
    configs = {
        "ultra_low_power": create_neuromorphic_config(
            power_budget_mw=25.0, 
            optimization_target="power"
        ),
        "balanced": create_neuromorphic_config(
            power_budget_mw=50.0,
            optimization_target="balanced"
        ),
        "high_speed": create_neuromorphic_config(
            power_budget_mw=100.0,
            optimization_target="speed"
        )
    }
    
    for name, config in configs.items():
        print(f"  {name}: {config.simulation_time}s sim, {config.power_budget_mw}mW budget")
    
    # Initialize neuromorphic FastVLM systems
    print("\n2. Initializing neuromorphic systems...")
    
    neuro_systems = {}
    for name, config in configs.items():
        neuro_systems[name] = NeuromorphicFastVLM(config)
        print(f"  ✓ {name} system initialized")
    
    # Create sample input features
    print("\n3. Creating sample input features...")
    
    # Simulated vision features (downsampled from 336x336x3)
    image_features = np.random.rand(1000) * 0.8  # Normalized features
    
    # Simulated text features (from BERT-like encoder)  
    text_features = np.random.rand(512) * 0.6
    
    print(f"  Vision features: {image_features.shape}")
    print(f"  Text features: {text_features.shape}")
    
    # Run inference on each system
    print("\n4. Running neuromorphic inference...")
    
    results = {}
    for name, system in neuro_systems.items():
        print(f"\n  Running {name} configuration:")
        
        start_time = time.time()
        output, metrics = system.inference(image_features, text_features)
        inference_time = time.time() - start_time
        
        results[name] = {
            "output_shape": output.shape,
            "inference_time": inference_time,
            "metrics": metrics
        }
        
        print(f"    Output shape: {output.shape}")
        print(f"    Inference time: {inference_time:.3f}s")
        print(f"    Energy consumed: {metrics['energy_consumed_mj']:.2f}mJ")
        print(f"    Power consumption: {metrics['vision_stats']['power_consumption_mw']:.1f}mW")
    
    # Compare performance
    print("\n5. Performance Comparison:")
    print("-" * 60)
    print(f"{'Configuration':<15} {'Time(s)':<8} {'Energy(mJ)':<12} {'Power(mW)':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        time_val = result["inference_time"]
        energy_val = result["metrics"]["energy_consumed_mj"]
        power_val = result["metrics"]["vision_stats"]["power_consumption_mw"]
        
        print(f"{name:<15} {time_val:<8.3f} {energy_val:<12.2f} {power_val:<10.1f}")
    
    # Demonstrate adaptive learning
    print("\n6. Adaptive Learning Demo:")
    
    balanced_system = neuro_systems["balanced"]
    
    # Simulate feedback for adaptation
    feedback = {"accuracy": 0.75}
    balanced_system.adapt_network(feedback)
    
    print("  ✓ Network adapted based on performance feedback")
    
    # Get comprehensive statistics
    print("\n7. Comprehensive Statistics:")
    
    stats = balanced_system.get_comprehensive_stats()
    print(f"  Total inferences: {stats['inference_count']}")
    print(f"  Average energy per inference: {stats['avg_energy_per_inference_mj']:.2f}mJ")
    
    # Network details
    vision_stats = stats['networks']['vision']
    print(f"  Vision network:")
    print(f"    Total neurons: {vision_stats['total_neurons']}")
    print(f"    Active neurons: {vision_stats['active_neurons']}")
    print(f"    Total spikes: {vision_stats['total_spikes']}")
    print(f"    Average firing rate: {vision_stats['avg_firing_rate_hz']:.2f}Hz")
    
    # Benchmark different configurations
    print("\n8. Running comprehensive benchmark...")
    
    benchmark_results = benchmark_neuromorphic_performance(
        image_features, 
        text_features,
        num_iterations=10
    )
    
    print("\nBenchmark Results:")
    for config_name, results in benchmark_results.items():
        print(f"\n  {config_name.upper()} Configuration:")
        print(f"    Avg inference time: {results['avg_inference_time_s']:.3f}s")
        print(f"    Energy efficiency: {results['energy_efficiency_fps_per_mw']:.1f} fps/mW")
        print(f"    Total energy: {results['total_energy_mj']:.2f}mJ")
    
    print("\n✨ Neuromorphic demo completed!")
    print("\nKey advantages of neuromorphic processing:")
    print("  • Ultra-low power consumption (10-100x less than traditional)")
    print("  • Event-driven processing (only compute when needed)")
    print("  • Adaptive learning and self-optimization")
    print("  • Excellent for battery-powered mobile devices")


if __name__ == "__main__":
    main()