# FractiVerse Testing Infrastructure

## Overview

The FractiVerse testing infrastructure is designed to be modular, extensible, and comprehensive. It covers multiple functional domains and uses pytest as the primary testing framework with additional support for visualization, metrics collection, and reporting.

## Test Organization

Tests are organized into functional domains, each with its own dedicated test module:

### Core System Tests
- `test_fractiverse.py` - Main application tests
- `test_api.py` - API endpoint tests
- `test_config.py` - Configuration management tests

### Mathematical Domain
- `test_math/`
  - `test_fractal_math.py` - Fractal mathematics operations
  - `test_transformations.py` - Geometric transformations
  - `test_field_equations.py` - Field equation solvers
  - `test_metrics.py` - Mathematical metrics and measurements

### Cognitive Engine Tests
- `test_cognitive/`
  - `test_learning.py` - Learning system tests
  - `test_decision.py` - Decision-making processes
  - `test_memory.py` - Memory management
  - `test_attention.py` - Attention mechanisms

### Unipixel Core Tests
- `test_unipixel/`
  - `test_core.py` - Core unipixel operations
  - `test_space.py` - Space management
  - `test_interactions.py` - Particle interactions
  - `test_fields.py` - Field behaviors

### FractiChain Tests
- `test_fractichain/`
  - `test_blockchain.py` - Blockchain operations
  - `test_consensus.py` - Consensus mechanisms
  - `test_smart_contracts.py` - Smart contract functionality
  - `test_network.py` - Network communication

### Visualization Tests
- `test_visualization/`
  - `test_renderers.py` - Rendering engines
  - `test_plots.py` - Plotting utilities
  - `test_animations.py` - Animation generation
  - `test_exports.py` - Export functionality

### Logging and Monitoring
- `test_logging.py` - Logging system tests
- `test_metrics.py` - Metrics collection and reporting
- `test_monitoring.py` - System monitoring

### Statistical Analysis
- `test_statistics/`
  - `test_analysis.py` - Statistical analysis tools
  - `test_distributions.py` - Probability distributions
  - `test_sampling.py` - Sampling methods
  - `test_inference.py` - Statistical inference

## Running Tests

### Basic Usage

```bash
# Run all tests
PYTHONPATH=. FRACTIVERSE_TESTING=true python3 -m pytest

# Run specific test module
pytest tests/test_math/test_fractal_math.py

# Run tests with coverage
pytest --cov=fractiverse tests/

# Run tests in parallel
pytest -n auto tests/
```

### Environment Variables

- `FRACTIVERSE_TESTING`: Set to "true" for test mode
- `FRACTIVERSE_PORT`: Custom port for testing (default: auto-assigned)
- `FRACTIVERSE_LOG_DIR`: Custom log directory
- `FRACTIVERSE_VIZ_DIR`: Custom visualization directory
- `FRACTIVERSE_METRICS_DIR`: Custom metrics directory

### Test Configuration

Tests can be configured through `conftest.py` and environment-specific configuration files:

```python
# conftest.py
@pytest.fixture(scope="session")
def test_config():
    return {
        "test_mode": True,
        "log_level": "DEBUG",
        "capture_traces": True
    }
```

## Test Output Structure

```
test_outputs/
├── logs/
│   ├── test_run.log
│   └── test_module_specific.log
├── visualizations/
│   ├── plots/
│   ├── animations/
│   └── diagrams/
├── metrics/
│   ├── performance_metrics.json
│   └── test_metrics.json
├── coverage/
│   └── htmlcov/
└── reports/
    ├── test_report.html
    └── test_summary.json
```

## Writing Tests

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **System Tests**: Test complete system functionality
4. **Performance Tests**: Test system performance and scalability
5. **Regression Tests**: Prevent reintroduction of fixed bugs

### Best Practices

1. **Test Independence**: Each test should be independent and self-contained
2. **Clear Naming**: Use descriptive test names that indicate functionality being tested
3. **Setup/Teardown**: Use fixtures for proper test setup and cleanup
4. **Assertions**: Use meaningful assertions and error messages
5. **Documentation**: Document test purpose and requirements

### Example Test Structure

```python
@pytest.mark.asyncio
async def test_cognitive_learning():
    """Test cognitive engine learning capabilities."""
    # Setup
    engine = CognitiveEngine()
    await engine.initialize()
    
    # Test learning process
    result = await engine.learn(test_data)
    assert result.success
    assert result.learning_rate > 0
    
    # Verify learned patterns
    patterns = engine.get_learned_patterns()
    assert len(patterns) > 0
    
    # Cleanup
    await engine.shutdown()
```

## Continuous Integration

Tests are automatically run in CI/CD pipeline:

1. On every push to main branch
2. On pull request creation/update
3. Daily scheduled runs for long-running tests

### CI Pipeline Stages

1. **Lint**: Code style and static analysis
2. **Unit Tests**: Fast, focused tests
3. **Integration Tests**: Component interaction tests
4. **System Tests**: Full system tests
5. **Performance Tests**: Benchmarking and performance validation

## Monitoring and Reporting

### Real-time Monitoring

- Test progress visualization
- Resource usage tracking
- Performance metrics collection

### Test Reports

- HTML test reports with interactive visualizations
- JSON metrics export
- Coverage reports
- Performance benchmarks

## Extending the Test Suite

### Adding New Test Modules

1. Create new test file in appropriate directory
2. Import required components
3. Define test fixtures if needed
4. Implement test cases
5. Add to test discovery path

### Creating Custom Fixtures

```python
@pytest.fixture(scope="module")
async def test_cognitive_engine():
    """Provide a test instance of the cognitive engine."""
    engine = CognitiveEngine(test_mode=True)
    await engine.initialize()
    yield engine
    await engine.shutdown()
```

## Future Improvements

1. **Automated Test Generation**: ML-based test case generation
2. **Chaos Testing**: Random failure injection
3. **Property-Based Testing**: Automated edge case discovery
4. **Visual Regression Testing**: UI/visualization comparison
5. **Performance Regression Detection**: Automated performance analysis

## Contributing

1. Follow test naming conventions
2. Add appropriate documentation
3. Include relevant fixtures
4. Update TEST_README.md as needed
5. Add test cases for new features 