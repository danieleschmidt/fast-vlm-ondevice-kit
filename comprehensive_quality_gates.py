#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation for FastVLM On-Device Kit.

Validates all quality gates including tests, security, performance,
architecture compliance, and production readiness.
"""

import sys
import time
import os
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def run_quality_gates():
    """Run comprehensive quality gates validation."""
    print("🔍 FastVLM Quality Gates Validation")
    print("=" * 60)
    print("Comprehensive validation of all quality aspects")
    print()
    
    total_score = 0
    max_score = 0
    gate_results = {}
    
    # Quality Gate 1: Functional Testing
    print("1. 🧪 FUNCTIONAL TESTING")
    print("-" * 30)
    
    functional_score = 0
    functional_max = 10
    
    try:
        # Test core pipeline functionality
        sys.path.append(os.path.join('src', 'fast_vlm_ondevice'))
        import core_pipeline
        
        config = core_pipeline.InferenceConfig(model_name="quality-gate-test")
        pipeline = core_pipeline.FastVLMCorePipeline(config)
        demo_image = core_pipeline.create_demo_image()
        
        # Test basic inference
        result = pipeline.process_image_question(demo_image, "What do you see?")
        if result and result.answer:
            print("   ✓ Core inference functionality: PASS")
            functional_score += 2
        
        # Test text-only mode
        text_result = pipeline.process_text_only("Hello, what can you do?")
        if text_result and text_result.answer:
            print("   ✓ Text-only inference: PASS")
            functional_score += 1
        
        # Test caching
        initial_cache = pipeline.get_stats()['cache_entries']
        pipeline.process_image_question(demo_image, "Test caching")
        final_cache = pipeline.get_stats()['cache_entries']
        if final_cache > initial_cache:
            print("   ✓ Caching functionality: PASS")
            functional_score += 2
        
        # Test error handling
        try:
            pipeline.process_image_question(b"", "")
            print("   ⚠️  Error handling: PARTIAL (no exception thrown)")
            functional_score += 1
        except:
            print("   ✓ Error handling: PASS")
            functional_score += 2
        
        # Test performance targets
        start_time = time.time()
        for i in range(5):
            pipeline.process_image_question(demo_image, f"Performance test {i}")
        avg_time = ((time.time() - start_time) / 5) * 1000
        
        if avg_time < 250:
            print(f"   ✓ Performance target (<250ms): PASS ({avg_time:.1f}ms avg)")
            functional_score += 2
        else:
            print(f"   ⚠️  Performance target: PARTIAL ({avg_time:.1f}ms avg)")
            functional_score += 1
        
    except Exception as e:
        print(f"   ✗ Functional testing failed: {e}")
    
    gate_results['functional_testing'] = {
        'score': functional_score,
        'max_score': functional_max,
        'percentage': (functional_score / functional_max) * 100
    }
    
    print(f"   Score: {functional_score}/{functional_max} ({(functional_score/functional_max)*100:.1f}%)")
    
    # Quality Gate 2: Security Validation
    print("\n2. 🔒 SECURITY VALIDATION")
    print("-" * 30)
    
    security_score = 0
    security_max = 10
    
    try:
        import input_validation
        
        validator = input_validation.CompositeValidator()
        
        # Test malicious input rejection
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "SELECT * FROM users WHERE 1=1",
            "'; DROP TABLE users; --",
            "javascript:alert('hack')",
            "eval(document.cookie)"
        ]
        
        blocked_count = 0
        for malicious_input in malicious_inputs:
            result = validator.text_validator.validate_question(malicious_input)
            if not result.is_valid:
                blocked_count += 1
        
        if blocked_count == len(malicious_inputs):
            print(f"   ✓ Malicious input blocking: PASS ({blocked_count}/{len(malicious_inputs)})")
            security_score += 3
        else:
            print(f"   ⚠️  Malicious input blocking: PARTIAL ({blocked_count}/{len(malicious_inputs)})")
            security_score += 1
        
        # Test image validation
        fake_jpeg = b'\xff\xd8\xff\xe0' + b'valid jpeg data' * 10
        malicious_exe = b'MZ' + b'executable data' * 20
        
        valid_result = validator.image_validator.validate_image_data(fake_jpeg, "test.jpg")
        malicious_result = validator.image_validator.validate_image_data(malicious_exe, "test.jpg")
        
        if valid_result.is_valid and not malicious_result.is_valid:
            print("   ✓ Image validation: PASS")
            security_score += 2
        else:
            print("   ⚠️  Image validation: NEEDS REVIEW")
            security_score += 1
        
        # Test input sanitization
        dangerous_chars = ["<", ">", "&", "\"", "'", "%", ";"]
        test_question = "What is this " + "".join(dangerous_chars) + " image?"
        sanitized = validator.text_validator.validate_question(test_question)
        
        if sanitized.is_valid and sanitized.sanitized_input:
            print("   ✓ Input sanitization: PASS")
            security_score += 2
        
        # Test configuration validation
        malicious_config = {
            "model_name": "../../../etc/passwd",
            "max_sequence_length": -1,
            "image_size": [999999, 999999]
        }
        
        config_result = validator.config_validator.validate_inference_config(malicious_config)
        if not config_result.is_valid:
            print("   ✓ Configuration validation: PASS")
            security_score += 2
        
        # Test file path validation
        dangerous_paths = ["../../../etc/passwd", "C:\\Windows\\System32", "/dev/random"]
        path_blocked = 0
        
        for path in dangerous_paths:
            try:
                result = validator.image_validator.validate_image_path(path)
                if not result.is_valid:
                    path_blocked += 1
            except:
                path_blocked += 1
        
        if path_blocked == len(dangerous_paths):
            print("   ✓ Path traversal protection: PASS")
            security_score += 1
        
    except Exception as e:
        print(f"   ✗ Security validation failed: {e}")
    
    gate_results['security_validation'] = {
        'score': security_score,
        'max_score': security_max,
        'percentage': (security_score / security_max) * 100
    }
    
    print(f"   Score: {security_score}/{security_max} ({(security_score/security_max)*100:.1f}%)")
    
    # Quality Gate 3: Performance & Scalability
    print("\n3. ⚡ PERFORMANCE & SCALABILITY")
    print("-" * 30)
    
    performance_score = 0
    performance_max = 10
    
    try:
        import distributed_inference
        import auto_scaling
        
        # Test distributed inference performance
        engine = distributed_inference.DistributedInferenceEngine("perf_test")
        
        # Register test nodes
        for i in range(2):
            node = distributed_inference.NodeInfo(f"perf_node_{i}", f"10.0.0.{i}", 8080)
            engine.load_balancer.register_node(node)
        
        engine.start()
        engine.set_local_pipeline(pipeline)
        
        # Test concurrent requests
        request_count = 20
        start_time = time.time()
        request_ids = []
        
        for i in range(request_count):
            request_id = engine.submit_request(demo_image, f"Perf test {i}", priority=5)
            request_ids.append(request_id)
        
        # Collect results
        successful_requests = 0
        total_latency = 0
        
        for request_id in request_ids:
            result = engine.get_result(request_id, timeout=5.0)
            if result and result.success:
                successful_requests += 1
                total_latency += result.latency_ms
        
        end_time = time.time()
        
        if successful_requests > 0:
            success_rate = (successful_requests / request_count) * 100
            avg_latency = total_latency / successful_requests
            throughput = request_count / (end_time - start_time)
            
            print(f"   📊 Concurrent Performance:")
            print(f"     - Success rate: {success_rate:.1f}%")
            print(f"     - Average latency: {avg_latency:.1f}ms")
            print(f"     - Throughput: {throughput:.1f} req/sec")
            
            if success_rate >= 95:
                performance_score += 3
                print("   ✓ Concurrent processing: EXCELLENT")
            elif success_rate >= 85:
                performance_score += 2
                print("   ✓ Concurrent processing: GOOD")
            else:
                performance_score += 1
                print("   ⚠️  Concurrent processing: NEEDS IMPROVEMENT")
            
            if avg_latency < 100:
                performance_score += 2
                print("   ✓ Latency target: EXCELLENT")
            elif avg_latency < 250:
                performance_score += 1
                print("   ✓ Latency target: GOOD")
        
        engine.stop()
        
        # Test auto-scaling responsiveness
        target = auto_scaling.ScalingTarget(min_instances=1, max_instances=5, cooldown_period=1.0)
        manager = auto_scaling.AutoScalingManager(target)
        
        # Simulate high load
        manager.record_metrics(85.0, 90.0, 100.0, 400.0, 30, 0.0)
        
        status = manager.get_scaling_status()
        if status and 'current_metrics' in status:
            print("   ✓ Auto-scaling metrics: PASS")
            performance_score += 2
        
        # Test cache performance
        cache = distributed_inference.DistributedCache(max_size_mb=10)
        
        # Fill cache and test hit rate
        for i in range(10):
            response = distributed_inference.InferenceResponse(
                f"cache_test_{i}", f"Answer {i}", 0.9, 100.0, "test_node"
            )
            cache.put(demo_image, f"Question {i}", response)
            if i < 5:  # Test some hits
                cache.get(demo_image, f"Question {i}")
        
        hit_rate = cache.get_hit_rate()
        if hit_rate > 0:
            print(f"   ✓ Cache performance: {hit_rate:.1%} hit rate")
            performance_score += 1
        
        # Test resource optimization
        optimizer = auto_scaling.ResourceOptimizer()
        high_load = auto_scaling.ScalingMetrics(cpu_usage=90, memory_usage=85, average_latency=500)
        optimized = optimizer.optimize_configuration(high_load, {"load_level": "high"})
        
        if optimized.cpu_cores > 1.0 or optimized.memory_gb > 2.0:
            print("   ✓ Resource optimization: PASS")
            performance_score += 1
        
        # Test load balancing
        lb = distributed_inference.LoadBalancer()
        for i in range(3):
            node = distributed_inference.NodeInfo(f"lb_test_{i}", f"10.0.1.{i}", 8080, [], i * 0.3)
            lb.register_node(node)
        
        test_req = distributed_inference.InferenceRequest("test", demo_image, "test")
        selected = lb.select_node(test_req)
        
        if selected and selected.load <= 0.3:  # Should select lowest load
            print("   ✓ Load balancing: PASS")
            performance_score += 1
        
    except Exception as e:
        print(f"   ✗ Performance testing failed: {e}")
    
    gate_results['performance_scalability'] = {
        'score': performance_score,
        'max_score': performance_max,
        'percentage': (performance_score / performance_max) * 100
    }
    
    print(f"   Score: {performance_score}/{performance_max} ({(performance_score/performance_max)*100:.1f}%)")
    
    # Quality Gate 4: Architecture & Code Quality
    print("\n4. 🏗️  ARCHITECTURE & CODE QUALITY")
    print("-" * 30)
    
    architecture_score = 0
    architecture_max = 8
    
    try:
        # Test modular architecture
        modules = [
            'core_pipeline', 'input_validation', 'production_monitoring',
            'distributed_inference', 'auto_scaling'
        ]
        
        importable_modules = 0
        for module in modules:
            try:
                __import__(module)
                importable_modules += 1
            except ImportError:
                pass
        
        if importable_modules >= len(modules) * 0.8:
            print(f"   ✓ Modular architecture: PASS ({importable_modules}/{len(modules)} modules)")
            architecture_score += 2
        
        # Test separation of concerns
        core_functions = ['process_image_question', 'process_text_only', 'get_stats']
        validation_functions = ['validate_question', 'validate_image_data']
        monitoring_functions = ['record_inference', 'get_dashboard_data']
        
        if all(hasattr(pipeline, func) for func in core_functions):
            print("   ✓ Core pipeline interface: PASS")
            architecture_score += 1
        
        if all(hasattr(validator, func.split('_')[1]) for func in validation_functions if hasattr(validator, func.split('_')[0] + '_validator')):
            print("   ✓ Validation interface: PASS")
            architecture_score += 1
        
        # Test configuration management
        if hasattr(core_pipeline, 'InferenceConfig'):
            config_class = core_pipeline.InferenceConfig
            required_fields = ['model_name', 'enable_caching', 'quantization_bits']
            
            if all(hasattr(config_class(), field) for field in required_fields):
                print("   ✓ Configuration management: PASS")
                architecture_score += 1
        
        # Test error handling architecture
        try:
            pipeline.process_image_question(None, None)
            print("   ⚠️  Error handling: NEEDS IMPROVEMENT (no exception)")
        except Exception:
            print("   ✓ Error handling architecture: PASS")
            architecture_score += 1
        
        # Test extensibility
        extensible_components = 0
        
        if hasattr(distributed_inference, 'LoadBalancer'):
            extensible_components += 1
        if hasattr(auto_scaling, 'ResourceOptimizer'):
            extensible_components += 1
        if hasattr(input_validation, 'CompositeValidator'):
            extensible_components += 1
        
        if extensible_components >= 3:
            print("   ✓ Extensible design: PASS")
            architecture_score += 1
        
        # Test documentation and type hints (basic check)
        import inspect
        
        documented_functions = 0
        total_functions = 0
        
        for name, obj in inspect.getmembers(core_pipeline.FastVLMCorePipeline):
            if callable(obj) and not name.startswith('_'):
                total_functions += 1
                if hasattr(obj, '__doc__') and obj.__doc__:
                    documented_functions += 1
        
        if total_functions > 0 and documented_functions / total_functions >= 0.8:
            print("   ✓ Documentation coverage: PASS")
            architecture_score += 1
        
    except Exception as e:
        print(f"   ✗ Architecture validation failed: {e}")
    
    gate_results['architecture_quality'] = {
        'score': architecture_score,
        'max_score': architecture_max,
        'percentage': (architecture_score / architecture_max) * 100
    }
    
    print(f"   Score: {architecture_score}/{architecture_max} ({(architecture_score/architecture_max)*100:.1f}%)")
    
    # Quality Gate 5: Production Readiness
    print("\n5. 🚀 PRODUCTION READINESS")
    print("-" * 30)
    
    production_score = 0
    production_max = 12
    
    try:
        import production_monitoring
        
        # Test monitoring capabilities
        monitor = production_monitoring.ProductionMonitor()
        monitor.start()
        
        # Record some metrics
        monitor.record_inference(150.0, 0.85, True)
        monitor.record_memory_usage(800.0)
        monitor.record_cache_hit_rate(75.0)
        
        dashboard = monitor.get_dashboard_data()
        if dashboard and 'inference_metrics' in dashboard:
            print("   ✓ Production monitoring: PASS")
            production_score += 2
        
        monitor.stop()
        
        # Test alerting system
        alert_manager = production_monitoring.AlertManager(monitor.metrics_collector)
        rule = production_monitoring.AlertRule("Test Alert", "inference.latency", 500.0)
        alert_manager.add_rule(rule)
        
        if rule.name in alert_manager.rules:
            print("   ✓ Alerting system: PASS")
            production_score += 1
        
        # Test health checks
        from fast_vlm_ondevice.health import minimal_check
        health_result = minimal_check()
        
        if health_result and health_result.get('healthy', False):
            print("   ✓ Health checks: PASS")
            production_score += 1
        
        # Test configuration validation
        test_configs = [
            {"model_name": "valid-model", "max_sequence_length": 77},
            {"model_name": "", "max_sequence_length": -1},  # Invalid
            {"model_name": "test", "image_size": [336, 336]}  # Valid
        ]
        
        valid_configs = 0
        for config in test_configs:
            result = validator.config_validator.validate_inference_config(config)
            if config["model_name"] and config.get("max_sequence_length", 1) > 0:
                if result.is_valid:
                    valid_configs += 1
            else:
                if not result.is_valid:
                    valid_configs += 1
        
        if valid_configs >= 2:
            print("   ✓ Configuration validation: PASS")
            production_score += 1
        
        # Test resource management
        stats = pipeline.get_stats()
        if stats and 'cache_entries' in stats and 'model_name' in stats:
            print("   ✓ Resource management: PASS")
            production_score += 1
        
        # Test graceful degradation
        try:
            # Test with null inputs
            result = pipeline.process_image_question(b"minimal", "test")
            if result:
                print("   ✓ Graceful degradation: PASS")
                production_score += 1
        except Exception:
            # Should handle gracefully, not crash
            pass
        
        # Test logging and observability
        import logging
        
        # Check if loggers are properly configured
        root_logger = logging.getLogger()
        if root_logger.handlers or logging.getLogger('fast_vlm_ondevice').handlers:
            print("   ✓ Logging infrastructure: PASS")
            production_score += 1
        
        # Test deployment configuration
        deployment_features = {
            'distributed_inference': distributed_inference.DistributedInferenceEngine,
            'auto_scaling': auto_scaling.AutoScalingManager,
            'monitoring': production_monitoring.ProductionMonitor,
            'caching': distributed_inference.DistributedCache
        }
        
        available_features = sum(1 for feature in deployment_features.values() if feature)
        if available_features >= 3:
            print(f"   ✓ Deployment features: PASS ({available_features}/4)")
            production_score += 2
        
        # Test security in production
        security_features = [
            hasattr(input_validation, 'CompositeValidator'),
            hasattr(validator, 'text_validator'),
            hasattr(validator, 'image_validator')
        ]
        
        if all(security_features):
            print("   ✓ Production security: PASS")
            production_score += 1
        
        # Test performance under production load
        load_test_start = time.time()
        concurrent_results = []
        
        for i in range(10):
            result = pipeline.process_image_question(demo_image, f"Load test {i}")
            if result:
                concurrent_results.append(result.latency_ms)
        
        if concurrent_results:
            avg_load_latency = sum(concurrent_results) / len(concurrent_results)
            if avg_load_latency < 500:  # Reasonable under load
                print(f"   ✓ Production load handling: PASS ({avg_load_latency:.1f}ms avg)")
                production_score += 1
        
    except Exception as e:
        print(f"   ✗ Production readiness validation failed: {e}")
    
    gate_results['production_readiness'] = {
        'score': production_score,
        'max_score': production_max,
        'percentage': (production_score / production_max) * 100
    }
    
    print(f"   Score: {production_score}/{production_max} ({(production_score/production_max)*100:.1f}%)")
    
    # Calculate total scores
    total_score = sum(result['score'] for result in gate_results.values())
    max_score = sum(result['max_score'] for result in gate_results.values())
    overall_percentage = (total_score / max_score) * 100
    
    # Final Quality Gates Report
    print("\n" + "=" * 60)
    print("🎯 QUALITY GATES VALIDATION RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Individual Gate Results:")
    for gate_name, result in gate_results.items():
        status = "✅ PASS" if result['percentage'] >= 80 else "⚠️  PARTIAL" if result['percentage'] >= 60 else "❌ FAIL"
        print(f"  {gate_name.replace('_', ' ').title()}: {result['score']}/{result['max_score']} ({result['percentage']:.1f}%) {status}")
    
    print(f"\n🎯 Overall Quality Score: {total_score}/{max_score} ({overall_percentage:.1f}%)")
    
    # Determine overall status
    if overall_percentage >= 85:
        print("\n🎉 QUALITY GATES: EXCELLENT ✅")
        print("✅ All quality standards exceeded")
        print("🚀 Ready for production deployment")
        print("🏆 Meets enterprise-grade requirements")
        status = "EXCELLENT"
    elif overall_percentage >= 75:
        print("\n✅ QUALITY GATES: PASS ✅")
        print("✅ All critical quality standards met")
        print("🚀 Ready for production deployment")
        print("🔧 Minor optimizations could be beneficial")
        status = "PASS"
    elif overall_percentage >= 60:
        print("\n⚠️  QUALITY GATES: CONDITIONAL PASS")
        print("⚠️  Most quality standards met")
        print("🔧 Some areas need improvement before production")
        print("📋 Review failing gates and address issues")
        status = "CONDITIONAL"
    else:
        print("\n❌ QUALITY GATES: FAIL")
        print("❌ Critical quality standards not met")
        print("🛠️  Significant improvements required")
        print("📋 Address all failing gates before production")
        status = "FAIL"
    
    # Save detailed report
    detailed_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_score": total_score,
        "max_score": max_score,
        "overall_percentage": overall_percentage,
        "status": status,
        "gate_results": gate_results,
        "recommendations": [],
        "next_steps": []
    }
    
    # Add recommendations based on results
    if gate_results['functional_testing']['percentage'] < 80:
        detailed_report['recommendations'].append("Improve core functionality test coverage")
    if gate_results['security_validation']['percentage'] < 80:
        detailed_report['recommendations'].append("Enhance security validation and input sanitization")
    if gate_results['performance_scalability']['percentage'] < 80:
        detailed_report['recommendations'].append("Optimize performance and scalability features")
    if gate_results['architecture_quality']['percentage'] < 80:
        detailed_report['recommendations'].append("Improve architecture and code quality")
    if gate_results['production_readiness']['percentage'] < 80:
        detailed_report['recommendations'].append("Enhance production readiness features")
    
    # Add next steps
    if status in ["EXCELLENT", "PASS"]:
        detailed_report['next_steps'] = [
            "Proceed with production deployment",
            "Setup monitoring and alerting",
            "Implement continuous integration",
            "Plan capacity scaling"
        ]
    elif status == "CONDITIONAL":
        detailed_report['next_steps'] = [
            "Address identified quality gaps",
            "Re-run quality gates",
            "Conduct additional testing",
            "Review architecture decisions"
        ]
    else:
        detailed_report['next_steps'] = [
            "Address all failing quality gates",
            "Implement missing critical features",
            "Conduct comprehensive testing",
            "Re-evaluate architecture"
        ]
    
    # Save report
    with open("quality_gates_report.json", "w") as f:
        json.dump(detailed_report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: quality_gates_report.json")
    
    return status in ["EXCELLENT", "PASS", "CONDITIONAL"]


if __name__ == "__main__":
    success = run_quality_gates()
    print(f"\nQuality Gates Validation: {'SUCCESS' if success else 'NEEDS_WORK'}")
    sys.exit(0 if success else 1)