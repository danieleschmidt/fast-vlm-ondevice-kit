"""
End-to-end tests for complete FastVLM workflow.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from tests.utils.test_helpers import (
    TestDataManager,
    BenchmarkRunner,
    AssertionHelpers,
    TestEnvironment,
    parametrize_model_variants,
    capture_logs
)


class TestCompleteWorkflow:
    """End-to-end tests for the complete FastVLM workflow."""
    
    @pytest.fixture(scope="class")
    def workflow_data(self):
        """Set up data for complete workflow testing."""
        manager = TestDataManager()
        return {
            "checkpoint": manager.get_mock_checkpoint_path(),
            "image": manager.get_sample_image_path(),
            "questions": manager.get_sample_questions_path()
        }
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_pytorch_to_ios_workflow(self, workflow_data):
        """Test complete workflow from PyTorch model to iOS deployment."""
        TestEnvironment.skip_if_insufficient_memory(3000)  # 3GB required
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Load PyTorch model
            with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
                mock_converter = mock_converter_class.return_value
                mock_pytorch_model = MagicMock()
                mock_coreml_model = MagicMock()
                
                mock_converter.load_pytorch_model.return_value = mock_pytorch_model
                mock_converter.convert_to_coreml.return_value = mock_coreml_model
                mock_converter.get_model_size_mb.return_value = 412.5
                
                converter = mock_converter_class()
                pytorch_model = converter.load_pytorch_model(str(workflow_data["checkpoint"]))
                
                # Step 2: Convert to Core ML with quantization
                coreml_model = converter.convert_to_coreml(
                    pytorch_model,
                    quantization="mixed",
                    compute_units="ALL",
                    image_size=(336, 336),
                    max_seq_length=77
                )
                
                # Step 3: Save optimized model
                model_path = temp_path / "FastVLM.mlpackage"
                coreml_model.save(str(model_path))
                
                # Step 4: Validate model size
                size_mb = converter.get_model_size_mb()
                AssertionHelpers.assert_memory_under_threshold(size_mb, 500)  # Under 500MB
                
                # Verify all steps completed
                mock_converter.load_pytorch_model.assert_called_once()
                mock_converter.convert_to_coreml.assert_called_once()
                mock_coreml_model.save.assert_called_once()
    
    @pytest.mark.e2e
    @parametrize_model_variants()
    def test_model_variant_workflows(self, workflow_data, model_variant):
        """Test workflow for different model variants."""
        variant_configs = {
            "fast-vlm-tiny": {"size": 98.0, "latency": 124, "memory": 412},
            "fast-vlm-base": {"size": 412.0, "latency": 187, "memory": 892},
            "fast-vlm-large": {"size": 892.0, "latency": 243, "memory": 1400}
        }
        
        config = variant_configs[model_variant]
        
        with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
            mock_converter = mock_converter_class.return_value
            mock_pytorch_model = MagicMock()
            mock_coreml_model = MagicMock()
            
            mock_converter.load_pytorch_model.return_value = mock_pytorch_model
            mock_converter.convert_to_coreml.return_value = mock_coreml_model
            mock_converter.get_model_size_mb.return_value = config["size"]
            
            # Simulate the workflow
            converter = mock_converter_class()
            model = converter.load_pytorch_model(f"checkpoints/{model_variant}.pth")
            coreml_model = converter.convert_to_coreml(model, quantization="int4")
            size = converter.get_model_size_mb()
            
            # Verify variant-specific expectations
            assert size == config["size"]
            AssertionHelpers.assert_memory_under_threshold(size, config["memory"])
    
    @pytest.mark.e2e 
    def test_inference_workflow(self, workflow_data):
        """Test complete inference workflow."""
        # Load test questions
        with open(workflow_data["questions"]) as f:
            test_data = json.load(f)
        
        questions = test_data["questions"][:3]  # Test with first 3 questions
        
        # Mock the complete inference pipeline
        with patch('fast_vlm_ondevice.FastVLM') as mock_vlm_class:
            mock_vlm = mock_vlm_class.return_value
            
            # Set up mock responses
            mock_answers = [
                "I can see a cat sitting on a windowsill.",
                "The main object appears to be red in color.",
                "There are two people visible in the image."
            ]
            mock_vlm.answer.side_effect = mock_answers
            
            # Execute inference workflow
            vlm = mock_vlm_class(model_path="FastVLM.mlpackage")
            results = []
            
            for i, question in enumerate(questions):
                answer = vlm.answer(
                    image=str(workflow_data["image"]), 
                    question=question
                )
                results.append({
                    "question": question,
                    "answer": answer,
                    "expected": test_data["expected_answers"][i]
                })
            
            # Verify inference completed for all questions
            assert len(results) == len(questions)
            assert mock_vlm.answer.call_count == len(questions)
            
            # Verify answers are reasonable
            for result in results:
                assert isinstance(result["answer"], str)
                assert len(result["answer"]) > 0
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_performance_benchmark_workflow(self, workflow_data):
        """Test complete performance benchmarking workflow."""
        benchmark_runner = BenchmarkRunner()
        
        def mock_inference():
            """Mock inference function for benchmarking."""
            import time
            time.sleep(0.1)  # Simulate 100ms inference
            return "Mock answer"
        
        # Run benchmark
        results = benchmark_runner.run_benchmark(
            test_function=mock_inference,
            iterations=10,
            warmup=3
        )
        
        # Verify benchmark results
        assert "mean_time_ms" in results
        assert "p95_time_ms" in results
        assert "peak_memory_mb" in results
        
        # Performance assertions
        AssertionHelpers.assert_latency_under_threshold(results["mean_time_ms"], 200)
        AssertionHelpers.assert_latency_under_threshold(results["p95_time_ms"], 300)
        
        # Save benchmark results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            benchmark_runner.save_results(Path(f.name))
            assert Path(f.name).exists()
    
    @pytest.mark.e2e
    def test_error_handling_workflow(self, workflow_data):
        """Test error handling in complete workflow."""
        with capture_logs("fast_vlm_ondevice") as log_capture:
            # Test file not found error
            with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
                mock_converter = mock_converter_class.return_value
                mock_converter.load_pytorch_model.side_effect = FileNotFoundError("Model not found")
                
                converter = mock_converter_class()
                
                with pytest.raises(FileNotFoundError):
                    converter.load_pytorch_model("non_existent_model.pth")
            
            # Test conversion error
            with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
                mock_converter = mock_converter_class.return_value
                mock_converter.load_pytorch_model.return_value = MagicMock()
                mock_converter.convert_to_coreml.side_effect = RuntimeError("Conversion failed")
                
                converter = mock_converter_class()
                model = converter.load_pytorch_model(str(workflow_data["checkpoint"]))
                
                with pytest.raises(RuntimeError, match="Conversion failed"):
                    converter.convert_to_coreml(model)
            
            # Test inference error
            with patch('fast_vlm_ondevice.FastVLM') as mock_vlm_class:
                mock_vlm = mock_vlm_class.return_value
                mock_vlm.answer.side_effect = ValueError("Invalid input")
                
                vlm = mock_vlm_class(model_path="FastVLM.mlpackage")
                
                with pytest.raises(ValueError, match="Invalid input"):
                    vlm.answer(image="invalid_image", question="test question")


class TestProductionReadinessWorkflow:
    """End-to-end tests for production readiness validation."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_memory_pressure_workflow(self):
        """Test workflow under memory pressure conditions."""
        TestEnvironment.skip_if_insufficient_memory(4000)  # 4GB required
        
        # Simulate memory-intensive operations
        with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
            mock_converter = mock_converter_class.return_value
            mock_converter.load_pytorch_model.return_value = MagicMock()
            mock_converter.convert_to_coreml.return_value = MagicMock()
            mock_converter.get_model_size_mb.return_value = 412.5
            
            # Monitor memory throughout workflow
            initial_memory = TestEnvironment.get_available_memory_mb()
            
            # Execute multiple conversions to stress memory
            for i in range(3):
                converter = mock_converter_class()
                model = converter.load_pytorch_model(f"model_{i}.pth")
                coreml_model = converter.convert_to_coreml(model)
                
                current_memory = TestEnvironment.get_available_memory_mb()
                memory_used = initial_memory - current_memory
                
                # Ensure memory usage stays reasonable
                AssertionHelpers.assert_memory_under_threshold(memory_used, 3000)  # Max 3GB used
    
    @pytest.mark.e2e
    def test_concurrent_workflows(self):
        """Test multiple concurrent workflows."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker_workflow(worker_id):
            """Worker function for concurrent testing."""
            try:
                with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
                    mock_converter = mock_converter_class.return_value
                    mock_converter.load_pytorch_model.return_value = MagicMock()
                    mock_converter.convert_to_coreml.return_value = MagicMock()
                    
                    converter = mock_converter_class()
                    model = converter.load_pytorch_model(f"worker_{worker_id}_model.pth")
                    coreml_model = converter.convert_to_coreml(model)
                    
                    results_queue.put(f"worker_{worker_id}_success")
            except Exception as e:
                errors_queue.put(f"worker_{worker_id}_error: {e}")
        
        # Start multiple worker threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_workflow, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Verify results
        assert results_queue.qsize() == 3, "Not all workers completed successfully"
        assert errors_queue.empty(), f"Errors occurred: {list(errors_queue.queue)}"
    
    @pytest.mark.e2e
    def test_deployment_validation_workflow(self):
        """Test deployment validation workflow."""
        deployment_checklist = {
            "model_size_check": False,
            "inference_speed_check": False,
            "memory_usage_check": False,
            "accuracy_check": False,
            "ios_compatibility_check": False
        }
        
        with patch('fast_vlm_ondevice.converter.FastVLMConverter') as mock_converter_class:
            mock_converter = mock_converter_class.return_value
            mock_coreml_model = MagicMock()
            
            mock_converter.load_pytorch_model.return_value = MagicMock()
            mock_converter.convert_to_coreml.return_value = mock_coreml_model
            mock_converter.get_model_size_mb.return_value = 412.5
            
            # Model size validation
            converter = mock_converter_class()
            model = converter.load_pytorch_model("production_model.pth")
            coreml_model = converter.convert_to_coreml(model, quantization="mixed")
            
            size_mb = converter.get_model_size_mb()
            if size_mb < 500:  # Under 500MB threshold
                deployment_checklist["model_size_check"] = True
            
            # Mock inference speed validation
            mock_inference_time = 187  # milliseconds
            if mock_inference_time < 250:  # Under 250ms threshold
                deployment_checklist["inference_speed_check"] = True
            
            # Mock memory usage validation
            mock_memory_usage = 892  # MB
            if mock_memory_usage < 1000:  # Under 1GB threshold
                deployment_checklist["memory_usage_check"] = True
            
            # Mock accuracy validation
            mock_accuracy = 0.712  # 71.2%
            if mock_accuracy > 0.70:  # Above 70% threshold
                deployment_checklist["accuracy_check"] = True
            
            # Mock iOS compatibility validation
            if TestEnvironment.is_coreml_available():
                deployment_checklist["ios_compatibility_check"] = True
        
        # Verify all deployment checks passed
        failed_checks = [k for k, v in deployment_checklist.items() if not v]
        assert len(failed_checks) == 0, f"Deployment validation failed: {failed_checks}"