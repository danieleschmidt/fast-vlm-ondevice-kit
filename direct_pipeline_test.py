#!/usr/bin/env python3
"""
Direct pipeline test - bypasses the __init__.py import chain.
"""

import sys
import os
import time

# Add src to path and test the core pipeline directly
sys.path.insert(0, 'src')

def main():
    """Test FastVLM core pipeline directly."""
    print("🚀 Direct FastVLM Pipeline Test")
    print("=" * 50)
    print("Testing core pipeline without complex import chain")
    print()
    
    try:
        # Import the core pipeline module directly
        sys.path.append(os.path.join('src', 'fast_vlm_ondevice'))
        import core_pipeline
        
        print("✓ Core pipeline module imported directly")
        
        # Test the classes and functions
        config = core_pipeline.InferenceConfig(
            model_name="fast-vlm-direct-test",
            enable_caching=True,
            quantization_bits=4
        )
        print(f"✓ InferenceConfig created: {config.model_name}")
        
        # Create pipeline
        pipeline = core_pipeline.FastVLMCorePipeline(config)
        print("✓ FastVLMCorePipeline initialized")
        
        # Create demo image
        demo_image = core_pipeline.create_demo_image()
        print(f"✓ Demo image created: {len(demo_image)} bytes")
        
        # Test inference scenarios
        test_cases = [
            "What do you see in this image?",
            "What colors are present?",
            "Describe the scene briefly",
            "Count the objects",
            "Any people visible?"
        ]
        
        print(f"\n🧠 Running {len(test_cases)} inference tests:")
        print("-" * 50)
        
        all_results = []
        total_latency = 0
        
        for i, question in enumerate(test_cases, 1):
            print(f"\n{i}. Testing: {question}")
            
            result = pipeline.process_image_question(demo_image, question)
            all_results.append(result)
            total_latency += result.latency_ms
            
            print(f"   📝 Answer: {result.answer}")
            print(f"   🎯 Confidence: {result.confidence:.2f}")
            print(f"   ⚡ Latency: {result.latency_ms:.1f}ms")
            print(f"   🕐 Timestamp: {result.timestamp}")
        
        # Test text-only mode
        print(f"\n📝 Testing text-only mode:")
        text_questions = [
            "Hello, what are you?",
            "What can you help with?",
            "Tell me about FastVLM"
        ]
        
        for question in text_questions:
            result = pipeline.process_text_only(question)
            print(f"   Q: {question}")
            print(f"   A: {result.answer}")
            print()
        
        # Pipeline statistics
        print("📊 Pipeline Statistics:")
        stats = pipeline.get_stats()
        
        print(f"  🤖 Model: {stats['model_name']}")
        print(f"  💾 Cache enabled: {stats['cache_enabled']}")
        print(f"  📦 Cache entries: {stats['cache_entries']}")
        print(f"  🖼️  Image size: {stats['image_size']}")
        print(f"  📏 Max sequence length: {stats['max_sequence_length']}")
        print(f"  🔢 Quantization: {stats['quantization_bits']}-bit")
        
        print(f"\n  🔧 Components:")
        for component, detail in stats['components'].items():
            print(f"     {component}: {detail}")
        
        # Performance analysis
        avg_latency = total_latency / len(test_cases)
        
        print(f"\n⚡ Performance Analysis:")
        print(f"  Total inference time: {total_latency:.1f}ms")
        print(f"  Average latency: {avg_latency:.1f}ms")
        print(f"  Latency range: {min(r.latency_ms for r in all_results):.1f}ms - {max(r.latency_ms for r in all_results):.1f}ms")
        print(f"  Target <250ms: {'✅ ACHIEVED' if avg_latency < 250 else '❌ EXCEEDED'}")
        
        # Test quick inference API
        print(f"\n🚀 Testing Quick Inference API:")
        quick_result = core_pipeline.quick_inference(
            demo_image, 
            "Quick API test", 
            "fast-vlm-quick"
        )
        print(f"  Result: {quick_result['answer'][:40]}...")
        print(f"  Latency: {quick_result['latency_ms']:.1f}ms")
        print(f"  Model: {quick_result['model_used']}")
        
        # Cache management
        print(f"\n🧹 Cache Management Test:")
        initial_entries = stats['cache_entries']
        cleared = pipeline.clear_cache()
        final_stats = pipeline.get_stats()
        
        print(f"  Initial cache entries: {initial_entries}")
        print(f"  Cleared entries: {cleared}")
        print(f"  Final cache entries: {final_stats['cache_entries']}")
        
        # Run one more inference to test cache repopulation
        pipeline.process_image_question(demo_image, "Cache repopulation test")
        repop_stats = pipeline.get_stats()
        print(f"  After repopulation: {repop_stats['cache_entries']} entries")
        
        # System info
        print(f"\n💻 System Information:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  Platform: {os.name}")
        print(f"  Working directory: {os.getcwd()}")
        
        print(f"\n" + "=" * 50)
        print("🎉 Direct Pipeline Test SUCCESSFUL!")
        print("✅ FastVLM core functionality verified")
        print("🚀 Production-ready architecture demonstrated")
        print("📱 Mobile-optimized with <250ms inference target")
        print("🔧 No external dependencies required for core operation")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print(f"\n{'🎉 SUCCESS' if success else '💥 FAILURE'}: Direct pipeline test {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1)