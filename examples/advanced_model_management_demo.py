#!/usr/bin/env python3
"""
Advanced Model Management Demo

Demonstrates comprehensive model lifecycle management, cross-platform support,
version control, and intelligent caching capabilities.
"""

import numpy as np
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fast_vlm_ondevice.model_manager import (
    ModelManager,
    ModelFormat,
    DeploymentTarget,
    create_advanced_model_manager
)
from src.fast_vlm_ondevice.neuromorphic import create_neuromorphic_config


def create_mock_model(model_type="pytorch"):
    """Create a mock model for demonstration."""
    if model_type == "pytorch":
        # Simplified model structure
        return {"weights": np.random.randn(1000, 512), "architecture": "FastVLM-Demo"}
    elif model_type == "neuromorphic":
        config = create_neuromorphic_config()
        return {"spike_config": config.__dict__, "model_type": "neuromorphic"}
    else:
        return {"mock_model": True, "type": model_type}


def main():
    """Run advanced model management demonstration."""
    
    print("🔧 Advanced Model Management Demo")
    print("=" * 50)
    
    # 1. Initialize Advanced Model Manager
    print("\n1. Initializing Advanced Model Manager...")
    
    model_manager = create_advanced_model_manager("demo_models")
    print("✅ Model manager initialized with cross-platform support")
    
    # 2. Save Models with Different Formats
    print("\n2. Saving Models with Different Formats...")
    
    model_ids = {}
    
    # Save PyTorch model
    pytorch_model = create_mock_model("pytorch")
    model_ids["pytorch"] = model_manager.save_model(
        pytorch_model,
        model_name="FastVLM-Mobile",
        version="1.0.0",
        model_format=ModelFormat.PYTORCH,
        target_devices=[DeploymentTarget.IOS, DeploymentTarget.ANDROID],
        quantization="int4",
        dependencies=["torch>=2.0.0", "transformers>=4.30.0"]
    )
    print(f"  ✅ PyTorch model saved: {model_ids['pytorch'][:12]}...")
    
    # Save neuromorphic variant
    neuromorphic_model = create_mock_model("neuromorphic")
    model_ids["neuromorphic"] = model_manager.save_model(
        neuromorphic_model,
        model_name="FastVLM-Neuromorphic",
        version="1.0.0",
        model_format=ModelFormat.NEUROMORPHIC,
        target_devices=[DeploymentTarget.NEUROMORPHIC_CHIP],
        quantization="spike",
        neuromorphic_config=neuromorphic_model,
        spike_encoding_type="rate",
        power_profile={"inference_mw": 25.0, "standby_mw": 0.1}
    )
    print(f"  ✅ Neuromorphic model saved: {model_ids['neuromorphic'][:12]}...")
    
    # Save optimized variant
    optimized_model = create_mock_model("pytorch")
    model_ids["optimized"] = model_manager.save_model(
        optimized_model,
        model_name="FastVLM-Optimized",
        version="1.1.0",
        model_format=ModelFormat.PYTORCH,
        target_devices=[DeploymentTarget.IOS],
        quantization="int8",
        parent_model_id=model_ids["pytorch"],
        derivation_method="quantization_optimization"
    )
    print(f"  ✅ Optimized model saved: {model_ids['optimized'][:12]}...")
    
    # 3. Model Registry Operations
    print("\n3. Model Registry Operations...")
    
    # List all models
    all_models = model_manager.registry.list_models()
    print(f"  Total models in registry: {len(all_models)}")
    
    # Search models
    mobile_models = model_manager.registry.search_models({
        "name_contains": "FastVLM",
        "max_size_mb": 1000,
        "quantization": "int4"
    })
    print(f"  Mobile-optimized models found: {len(mobile_models)}")
    
    # Get model versions
    versions = model_manager.registry.get_model_versions("FastVLM-Mobile")
    print(f"  FastVLM-Mobile versions: {len(versions)}")
    
    # 4. Model Loading and Caching
    print("\n4. Model Loading and Caching...")
    
    # Load models (demonstrates caching)
    print("  Loading PyTorch model...")
    start_time = time.time()
    pytorch_loaded = model_manager.load_model(model_ids["pytorch"])
    load_time = time.time() - start_time
    print(f"    First load time: {load_time:.3f}s")
    
    # Load again (from cache)
    start_time = time.time()
    pytorch_cached = model_manager.load_model(model_ids["pytorch"])
    cache_time = time.time() - start_time
    print(f"    Cached load time: {cache_time:.3f}s ({load_time/cache_time:.1f}x faster)")
    
    # Load neuromorphic model
    print("  Loading neuromorphic model...")
    neuromorphic_loaded = model_manager.load_model(model_ids["neuromorphic"])
    print("    ✅ Neuromorphic model loaded successfully")
    
    # 5. Model Benchmarking
    print("\n5. Comprehensive Model Benchmarking...")
    
    for model_name, model_id in model_ids.items():
        print(f"\n  Benchmarking {model_name} model...")
        
        # Create mock test data
        test_data = {"images": np.random.randn(10, 3, 224, 224)}
        
        benchmark_results = model_manager.benchmark_model(
            model_id,
            test_data=test_data,
            metrics=["latency", "memory", "accuracy", "throughput"]
        )
        
        print(f"    Latency: {benchmark_results['latency_ms']:.1f}ms")
        print(f"    Memory: {benchmark_results['memory_mb']:.1f}MB")
        print(f"    Accuracy: {benchmark_results['accuracy']:.1%}")
        print(f"    Throughput: {benchmark_results['throughput_fps']:.1f} FPS")
        
        if model_name == "neuromorphic":
            print(f"    Energy: {benchmark_results.get('energy_mj', 0):.2f}mJ")
    
    # 6. Model Variants and Lineage
    print("\n6. Model Variants and Lineage...")
    
    # Create a variant using transformation
    def quantization_transform(model, target_bits=4):
        """Mock quantization transformation."""
        print(f"    Applying {target_bits}-bit quantization...")
        quantized = model.copy() if isinstance(model, dict) else model
        return quantized
    
    variant_id = model_manager.create_model_variant(
        base_model_id=model_ids["pytorch"],
        variant_name="FastVLM-Mobile-Quantized",
        derivation_method="aggressive_quantization",
        transformation_func=quantization_transform,
        target_bits=2
    )
    print(f"  ✅ Model variant created: {variant_id[:12]}...")
    
    # Show model lineage
    lineage = model_manager.registry.create_model_lineage(variant_id)
    print(f"  Model lineage:")
    if lineage["parent"]:
        print(f"    Parent: {lineage['parent']['model_name']}")
    print(f"    Current: {lineage['model']['model_name']}")
    print(f"    Children: {len(lineage['children'])}")
    
    # 7. Cross-Platform Deployment
    print("\n7. Cross-Platform Deployment...")
    
    # Export model package for deployment
    package_path = model_manager.export_model_package(
        model_ids["pytorch"],
        output_path="deployment_packages",
        include_dependencies=True
    )
    print(f"  ✅ Deployment package exported: {Path(package_path).name}")
    
    # Simulate loading for different target devices
    target_devices = [DeploymentTarget.IOS, DeploymentTarget.ANDROID]
    
    for target in target_devices:
        print(f"  Loading model optimized for {target.value}...")
        device_model = model_manager.load_model(model_ids["pytorch"], target_device=target)
        print(f"    ✅ Model loaded and optimized for {target.value}")
    
    # 8. Performance Analytics
    print("\n8. Performance Analytics...")
    
    # Get comprehensive statistics
    for model_name, model_id in model_ids.items():
        metadata = model_manager.registry.get_model(model_id)
        if metadata and metadata.benchmark_results:
            print(f"\n  {model_name.upper()} Model Analytics:")
            print(f"    Model size: {metadata.file_size_mb:.1f}MB")
            print(f"    Target devices: {[d.value for d in metadata.target_devices]}")
            print(f"    Quantization: {metadata.quantization}")
            
            # Show benchmark history
            recent_benchmarks = list(metadata.benchmark_results.keys())[-2:]
            print(f"    Recent benchmark runs: {len(recent_benchmarks)}")
            
            if recent_benchmarks:
                latest = metadata.benchmark_results[recent_benchmarks[-1]]
                print(f"    Latest performance:")
                for metric, value in latest.items():
                    if isinstance(value, (int, float)):
                        print(f"      {metric}: {value:.2f}")
    
    # 9. Model Registry Search and Filtering
    print("\n9. Advanced Model Search and Filtering...")
    
    # Search by performance criteria
    high_performance_models = model_manager.registry.search_models({
        "min_accuracy": 0.8,
        "max_latency_ms": 200
    })
    print(f"  High-performance models (>80% acc, <200ms): {len(high_performance_models)}")
    
    # Filter by deployment target
    ios_models = model_manager.registry.list_models(
        target_device=DeploymentTarget.IOS
    )
    print(f"  iOS-compatible models: {len(ios_models)}")
    
    # Filter by format
    neuromorphic_models = model_manager.registry.list_models(
        model_format=ModelFormat.NEUROMORPHIC
    )
    print(f"  Neuromorphic models: {len(neuromorphic_models)}")
    
    # 10. Cleanup and Summary
    print("\n10. System Summary and Cleanup...")
    
    cache_info = {
        "cached_models": len(model_manager.loaded_models),
        "total_registry_models": len(model_manager.registry.models),
        "cache_hits": sum(model_manager.model_access_counts.values())
    }
    
    print(f"  Registry models: {cache_info['total_registry_models']}")
    print(f"  Cached models: {cache_info['cached_models']}")
    print(f"  Total cache hits: {cache_info['cache_hits']}")
    
    # Shutdown model manager
    model_manager.shutdown()
    print("  ✅ Model manager shutdown complete")
    
    print("\n✨ Advanced model management demo completed!")
    print("\nKey capabilities demonstrated:")
    print("  🔧 Multi-format model support (PyTorch, CoreML, ONNX, Neuromorphic)")
    print("  🌐 Cross-platform deployment (iOS, Android, Linux, etc.)")
    print("  📊 Comprehensive benchmarking and analytics")
    print("  🧬 Model lineage and variant tracking")
    print("  ⚡ Intelligent caching with LRU eviction")
    print("  📦 Complete deployment package generation")
    print("  🔍 Advanced search and filtering")
    print("  🎯 Device-specific optimization")
    print("  📈 Performance monitoring and history")
    print("  🔒 Secure file handling and validation")


if __name__ == "__main__":
    main()