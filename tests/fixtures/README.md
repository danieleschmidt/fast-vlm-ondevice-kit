# Test Fixtures

This directory contains test data and fixtures for the FastVLM test suite.

## Structure

- `sample_image.jpg` - Sample test image for model inference testing
- `sample_questions.json` - Collection of test questions and expected answers
- `mock_checkpoint.pth` - Mock model checkpoint for testing conversion pipeline
- `benchmark_data/` - Performance benchmark datasets
- `integration_data/` - Integration test datasets

## Usage

Test fixtures are automatically generated when needed by the test utilities. 
Manual fixtures can be added here for specific test scenarios.

## Guidelines

- Keep fixture files small (<10MB total)
- Use representative but synthetic data when possible
- Document any real data sources and licensing
- Include both positive and negative test cases
- Ensure fixtures work across different platforms

## File Types

- **Images**: JPG format, 336x336 resolution preferred
- **Questions**: JSON format with question/answer pairs
- **Models**: Mock PyTorch checkpoint format (.pth)
- **Configs**: JSON/YAML configuration files