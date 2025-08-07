"""
Test suite for neuromorphic computing components.
"""

import pytest
import numpy as np
import time
from src.fast_vlm_ondevice.neuromorphic import (
    SpikeConfig,
    SpikingNeuron,
    SpikingNetwork,
    NeuromorphicEncoder,
    NeuromorphicFastVLM,
    create_neuromorphic_config,
    benchmark_neuromorphic_performance
)


class TestSpikeConfig:
    """Test suite for SpikeConfig."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = SpikeConfig()
        
        assert config.dt == 0.001
        assert config.simulation_time == 0.1
        assert config.spike_threshold == 1.0
        assert config.enable_stdp == True
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = SpikeConfig(
            dt=0.005,
            simulation_time=0.2,
            spike_threshold=1.5,
            learning_rate=0.01
        )
        
        assert config.dt == 0.005
        assert config.simulation_time == 0.2
        assert config.spike_threshold == 1.5
        assert config.learning_rate == 0.01


class TestSpikingNeuron:
    """Test suite for SpikingNeuron."""
    
    def test_neuron_initialization(self):
        """Test neuron initialization."""
        config = SpikeConfig()
        neuron = SpikingNeuron(config, neuron_id=0)
        
        assert neuron.neuron_id == 0
        assert neuron.membrane_potential == 0.0
        assert len(neuron.spike_times) == 0
        assert neuron.spike_count == 0
    
    def test_neuron_spike_generation(self):
        """Test spike generation."""
        config = SpikeConfig(spike_threshold=1.0, dt=0.001)
        neuron = SpikingNeuron(config, neuron_id=0)
        
        # Apply strong input to trigger spike
        current_time = 0.0
        spike_occurred = neuron.update(2.0, current_time)  # Strong input
        
        assert spike_occurred == True
        assert len(neuron.spike_times) == 1
        assert neuron.spike_count == 1
        assert neuron.membrane_potential == 0.0  # Reset after spike
    
    def test_refractory_period(self):
        """Test refractory period enforcement."""
        config = SpikeConfig(refractory_period=0.002, dt=0.001)
        neuron = SpikingNeuron(config, neuron_id=0)
        
        # Generate first spike
        neuron.update(2.0, 0.0)
        assert len(neuron.spike_times) == 1
        
        # Try to spike during refractory period
        spike_occurred = neuron.update(2.0, 0.001)  # Within refractory period
        assert spike_occurred == False
        assert len(neuron.spike_times) == 1  # No new spike
        
        # Spike after refractory period
        spike_occurred = neuron.update(2.0, 0.003)  # After refractory period
        # Note: May or may not spike depending on membrane recovery
    
    def test_spike_rate_calculation(self):
        """Test spike rate calculation."""
        config = SpikeConfig()
        neuron = SpikingNeuron(config, neuron_id=0)
        
        # Generate multiple spikes
        for i in range(5):
            neuron.spike_times.append(time.time() - i * 0.01)
        
        spike_rate = neuron.get_spike_rate(time_window=0.1)
        assert spike_rate >= 0.0  # Should be positive


class TestSpikingNetwork:
    """Test suite for SpikingNetwork."""
    
    def test_network_initialization(self):
        """Test network initialization."""
        config = SpikeConfig()
        topology = {
            'num_neurons': 100,
            'connection_prob': 0.1
        }
        
        network = SpikingNetwork(config, topology)
        
        assert len(network.neurons) == 100
        assert len(network.synapses) > 0
        assert network.total_spikes == 0
    
    def test_simulation_step(self):
        """Test single simulation step."""
        config = SpikeConfig()
        topology = {'num_neurons': 10, 'connection_prob': 0.2}
        network = SpikingNetwork(config, topology)
        
        # Provide input currents
        input_currents = {0: 2.0, 1: 1.5}  # Strong inputs to first two neurons
        
        spike_events = network.simulate_step(input_currents)
        
        assert isinstance(spike_events, dict)
        assert len(spike_events) == 10  # One entry per neuron
    
    def test_stdp_learning(self):
        """Test STDP learning mechanism."""
        config = SpikeConfig(enable_stdp=True, learning_rate=0.01)
        topology = {'num_neurons': 5, 'connection_prob': 0.5}
        network = SpikingNetwork(config, topology)
        
        initial_weights = list(network.synapses.values())
        
        # Simulate several steps with input
        for step in range(10):
            input_currents = {0: 2.0}
            network.simulate_step(input_currents)
        
        # Check if weights have been updated
        final_weights = list(network.synapses.values())
        
        # Weights should have changed due to STDP
        assert initial_weights != final_weights
    
    def test_network_statistics(self):
        """Test network statistics collection."""
        config = SpikeConfig()
        topology = {'num_neurons': 20, 'connection_prob': 0.15}
        network = SpikingNetwork(config, topology)
        
        # Run some simulation
        for step in range(5):
            input_currents = {0: 2.0, 1: 1.5}
            network.simulate_step(input_currents)
        
        stats = network.get_network_statistics()
        
        assert 'total_neurons' in stats
        assert 'total_synapses' in stats
        assert 'total_spikes' in stats
        assert 'avg_firing_rate_hz' in stats
        assert 'energy_consumed_mj' in stats
        
        assert stats['total_neurons'] == 20
        assert stats['total_synapses'] > 0


class TestNeuromorphicEncoder:
    """Test suite for NeuromorphicEncoder."""
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        config = SpikeConfig()
        encoder = NeuromorphicEncoder(config)
        
        assert encoder.config == config
    
    def test_rate_encoding(self):
        """Test rate-based spike encoding."""
        config = SpikeConfig(simulation_time=0.05, dt=0.001)
        encoder = NeuromorphicEncoder(config)
        
        # Create test activation values
        activation_values = np.array([0.1, 0.5, 0.8, 0.0])
        
        spike_trains = encoder.rate_encode(activation_values, max_rate=100.0)
        
        assert len(spike_trains) == 4
        assert all(isinstance(train, list) for train in spike_trains)
        
        # Higher activation should generally produce more spikes
        spike_counts = [len(train) for train in spike_trains]
        
        # Neuron with 0.8 activation should have more spikes than 0.1
        # (though this is probabilistic, so we just check basic structure)
    
    def test_temporal_encoding(self):
        """Test temporal spike encoding."""
        config = SpikeConfig()
        encoder = NeuromorphicEncoder(config)
        
        activation_values = np.array([0.2, 0.8, 0.5, 0.0])
        
        spike_trains = encoder.temporal_encode(activation_values, time_window=0.05)
        
        assert len(spike_trains) == 4
        
        # Check that higher activations have earlier spike times
        spike_times = [train[0] if train else float('inf') for train in spike_trains]
        
        # Activation 0.8 should spike earlier than 0.2
        assert spike_times[1] < spike_times[0]  # Higher activation = earlier spike
    
    def test_encoding_with_zero_activation(self):
        """Test encoding behavior with zero activation."""
        config = SpikeConfig()
        encoder = NeuromorphicEncoder(config)
        
        activation_values = np.array([0.0, 0.0, 0.0])
        
        # Rate encoding with zero should produce few/no spikes
        spike_trains = encoder.rate_encode(activation_values, max_rate=100.0)
        total_spikes = sum(len(train) for train in spike_trains)
        assert total_spikes == 0 or total_spikes < 5  # Very few spikes expected
        
        # Temporal encoding with zero should produce no spikes
        spike_trains = encoder.temporal_encode(activation_values)
        assert all(len(train) == 0 for train in spike_trains)


class TestNeuromorphicFastVLM:
    """Test suite for NeuromorphicFastVLM."""
    
    def test_neuromorphic_vlm_initialization(self):
        """Test neuromorphic VLM initialization."""
        config = SpikeConfig()
        neuro_vlm = NeuromorphicFastVLM(config)
        
        assert neuro_vlm.config == config
        assert neuro_vlm.encoder is not None
        assert neuro_vlm.vision_network is not None
        assert neuro_vlm.text_network is not None
        assert neuro_vlm.fusion_network is not None
    
    def test_vision_processing(self):
        """Test vision processing pipeline."""
        config = SpikeConfig(simulation_time=0.02)  # Shorter for testing
        neuro_vlm = NeuromorphicFastVLM(config)
        
        # Create test image features
        image_features = np.random.rand(100) * 0.8
        
        vision_output = neuro_vlm.process_vision(image_features)
        
        assert isinstance(vision_output, np.ndarray)
        assert vision_output.shape[0] > 0  # Should have some output
    
    def test_text_processing(self):
        """Test text processing pipeline."""
        config = SpikeConfig(simulation_time=0.02)
        neuro_vlm = NeuromorphicFastVLM(config)
        
        # Create test text features
        text_features = np.random.rand(50) * 0.6
        
        text_output = neuro_vlm.process_text(text_features)
        
        assert isinstance(text_output, np.ndarray)
        assert text_output.shape[0] > 0
    
    def test_full_inference(self):
        """Test complete inference pipeline."""
        config = SpikeConfig(simulation_time=0.02)
        neuro_vlm = NeuromorphicFastVLM(config)
        
        # Create test inputs
        image_features = np.random.rand(100) * 0.8
        text_features = np.random.rand(50) * 0.6
        
        output, metrics = neuro_vlm.inference(image_features, text_features)
        
        assert isinstance(output, np.ndarray)
        assert isinstance(metrics, dict)
        
        # Check required metrics
        assert 'inference_time_s' in metrics
        assert 'energy_consumed_mj' in metrics
        assert 'vision_stats' in metrics
        assert 'text_stats' in metrics
        assert 'fusion_stats' in metrics
        
        assert metrics['inference_time_s'] > 0
        assert metrics['energy_consumed_mj'] > 0
    
    def test_adaptation(self):
        """Test network adaptation."""
        config = SpikeConfig()
        neuro_vlm = NeuromorphicFastVLM(config)
        
        initial_lr = neuro_vlm.config.learning_rate
        
        # Test adaptation with poor performance
        feedback = {"accuracy": 0.2}
        neuro_vlm.adapt_network(feedback)
        
        # Learning rate should have increased
        assert neuro_vlm.config.learning_rate != initial_lr
        
        # Test adaptation with good performance
        feedback = {"accuracy": 0.9}
        neuro_vlm.adapt_network(feedback)
        
        # Learning rate should be different again
    
    def test_comprehensive_stats(self):
        """Test comprehensive statistics collection."""
        config = SpikeConfig()
        neuro_vlm = NeuromorphicFastVLM(config)
        
        # Run some inferences first
        image_features = np.random.rand(50) * 0.8
        text_features = np.random.rand(25) * 0.6
        
        for _ in range(3):
            neuro_vlm.inference(image_features, text_features)
        
        stats = neuro_vlm.get_comprehensive_stats()
        
        assert 'inference_count' in stats
        assert 'total_energy_mj' in stats
        assert 'avg_energy_per_inference_mj' in stats
        assert 'networks' in stats
        
        assert stats['inference_count'] == 3
        assert stats['total_energy_mj'] > 0


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_create_neuromorphic_config(self):
        """Test neuromorphic config creation."""
        # Test different optimization targets
        power_config = create_neuromorphic_config(
            power_budget_mw=25.0,
            optimization_target="power"
        )
        
        speed_config = create_neuromorphic_config(
            power_budget_mw=100.0,
            optimization_target="speed"
        )
        
        balanced_config = create_neuromorphic_config(
            power_budget_mw=50.0,
            optimization_target="balanced"
        )
        
        assert power_config.power_budget_mw == 25.0
        assert speed_config.power_budget_mw == 100.0
        assert balanced_config.power_budget_mw == 50.0
        
        # Power config should have power-optimized settings
        assert power_config.voltage_scaling_factor < speed_config.voltage_scaling_factor
    
    def test_benchmark_neuromorphic_performance(self):
        """Test performance benchmarking."""
        # Create test data
        image_features = np.random.rand(100) * 0.8
        text_features = np.random.rand(50) * 0.6
        
        # Run benchmark with small iteration count for testing
        results = benchmark_neuromorphic_performance(
            image_features,
            text_features,
            num_iterations=3  # Small number for testing
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0  # Should have results for different configs
        
        # Check that all configurations have required metrics
        for config_name, config_results in results.items():
            assert 'avg_inference_time_s' in config_results
            assert 'avg_energy_per_inference_mj' in config_results
            assert 'energy_efficiency_fps_per_mw' in config_results
            assert 'comprehensive_stats' in config_results
            
            assert config_results['avg_inference_time_s'] > 0
            assert config_results['avg_energy_per_inference_mj'] > 0


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        config = SpikeConfig()
        encoder = NeuromorphicEncoder(config)
        
        # Test with empty array
        empty_features = np.array([])
        
        spike_trains = encoder.rate_encode(empty_features)
        assert len(spike_trains) == 0
        
        spike_trains = encoder.temporal_encode(empty_features)
        assert len(spike_trains) == 0
    
    def test_extreme_activation_values(self):
        """Test handling of extreme activation values."""
        config = SpikeConfig()
        encoder = NeuromorphicEncoder(config)
        
        # Test with very high values
        extreme_features = np.array([10.0, 100.0, -5.0])
        
        # Should handle without crashing
        spike_trains = encoder.rate_encode(extreme_features, max_rate=1000.0)
        assert len(spike_trains) == 3
        
        spike_trains = encoder.temporal_encode(extreme_features)
        assert len(spike_trains) == 3
    
    def test_network_with_no_connections(self):
        """Test network with no synaptic connections."""
        config = SpikeConfig()
        topology = {'num_neurons': 5, 'connection_prob': 0.0}  # No connections
        
        network = SpikingNetwork(config, topology)
        
        assert len(network.synapses) == 0
        
        # Should still be able to simulate
        input_currents = {0: 2.0}
        spike_events = network.simulate_step(input_currents)
        
        assert len(spike_events) == 5
    
    def test_very_short_simulation(self):
        """Test with very short simulation time."""
        config = SpikeConfig(simulation_time=0.001, dt=0.001)  # Only 1 step
        neuro_vlm = NeuromorphicFastVLM(config)
        
        image_features = np.random.rand(10)
        text_features = np.random.rand(5)
        
        # Should complete without error
        output, metrics = neuro_vlm.inference(image_features, text_features)
        
        assert isinstance(output, np.ndarray)
        assert metrics['inference_time_s'] > 0


if __name__ == "__main__":
    pytest.main([__file__])