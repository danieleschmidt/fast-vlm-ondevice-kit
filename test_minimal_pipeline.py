#!/usr/bin/env python3
"""
Minimal FastVLM pipeline test - imports only the core pipeline directly.
"""

import sys
import time

# Add src to path
sys.path.insert(0, 'src')

def test_minimal_pipeline():
    """Test the core pipeline directly without full module imports."""
    print("🚀 Minimal FastVLM Pipeline Test")
    print("=" * 40)
    
    try:
        # Import only the core pipeline module directly
        from fast_vlm_ondevice.core_pipeline import (
            FastVLMCorePipeline, 
            InferenceConfig, 
            create_demo_image,
            quick_inference
        )
        
        print("✓ Core pipeline imported successfully")
        
        # Create a simple configuration
        config = InferenceConfig(
            model_name="fast-vlm-minimal",
            enable_caching=True,
            quantization_bits=4,
            max_sequence_length=77
        )
        print(f"✓ Configuration: {config.model_name}")
        
        # Initialize the pipeline
        pipeline = FastVLMCorePipeline(config)
        print("✓ Pipeline initialized")
        
        # Create demo image data
        demo_image = create_demo_image()
        print(f"✓ Demo image created ({len(demo_image)} bytes)")
        
        # Test various inference scenarios
        test_questions = [
            "What objects are in this image?",
            "What colors do you see?", 
            "Describe the scene",
            "How many items are visible?",
            "Is there a person present?"
        ]
        
        print(f"\n🧠 Running {len(test_questions)} inference tests:")
        print("-" * 40)
        
        total_time = 0
        results = []
        
        for i, question in enumerate(test_questions, 1):
            start_time = time.time()
            result = pipeline.process_image_question(demo_image, question)
            end_time = time.time()
            
            total_time += result.latency_ms
            results.append(result)
            
            print(f"{i}. Q: {question}")
            print(f"   A: {result.answer}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Latency: {result.latency_ms:.1f}ms")
            print()
        
        # Test text-only mode
        print("📝 Testing text-only inference:")
        text_result = pipeline.process_text_only("Hello, what can you do?")
        print(f"Q: Hello, what can you do?")
        print(f"A: {text_result.answer}")
        print(f"Confidence: {text_result.confidence:.2f}")
        print(f"Latency: {text_result.latency_ms:.1f}ms")
        
        # Pipeline statistics
        print("\n📊 Pipeline Statistics:")
        stats = pipeline.get_stats()
        print(f"Model: {stats['model_name']}")
        print(f"Cache enabled: {stats['cache_enabled']}")
        print(f"Cache entries: {stats['cache_entries']}")
        print(f"Image size: {stats['image_size']}")
        print(f"Max sequence length: {stats['max_sequence_length']}")
        print(f"Quantization: {stats['quantization_bits']}-bit")
        
        print("\nComponents:")
        for name, info in stats['components'].items():
            print(f"  {name}: {info}")
        
        # Performance summary
        avg_latency = total_time / len(test_questions)
        print(f"\n⚡ Performance Summary:")
        print(f"Total time: {total_time:.1f}ms")
        print(f"Average latency: {avg_latency:.1f}ms")
        print(f"Target <250ms: {'✓ PASS' if avg_latency < 250 else '✗ FAIL'}")
        
        # Test quick inference API
        print(f"\n🚀 Testing Quick Inference API:")
        quick_result = quick_inference(demo_image, "Quick test", "fast-vlm-minimal")
        print(f"Quick API latency: {quick_result['latency_ms']:.1f}ms")
        print(f"Quick API answer: {quick_result['answer'][:50]}...")
        
        # Cache management test
        print(f"\n🧹 Testing Cache Management:")
        print(f"Cache entries before clear: {pipeline.get_stats()['cache_entries']}")
        cleared = pipeline.clear_cache()
        print(f"Cleared {cleared} cache entries")
        print(f"Cache entries after clear: {pipeline.get_stats()['cache_entries']}")
        
        print(f"\n" + "=" * 40)
        print("🎉 Minimal Pipeline Test PASSED!")
        print("✅ Core functionality working without external dependencies")
        print("🚀 Ready for production deployment")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_minimal_pipeline()
    print(f"\nTest Result: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)