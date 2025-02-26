# FractiVerse 1.0

A fractal intelligence system with cognitive processing, visualization, and monitoring capabilities.

## Setup

Run tests with

FRACTIVERSE_TESTING=true python3 main.py 

From top level of the repo. 

## Running Tests

The test suite includes unit tests, integration tests, and system tests with comprehensive logging and visualization outputs.

### Basic Test Run

Run the entire test suite with verbose output and logging:
```bash
# Set testing environment
export FRACTIVERSE_TESTING=true  # Linux/Mac
# or
set FRACTIVERSE_TESTING=true  # Windows

# Run tests
python3 -m pytest tests/ -v --capture=no
```

### Advanced Test Options

1. Run tests with coverage:
```bash
python3 -m pytest tests/ --cov=fractiverse --cov-report=html
```

2. Run tests in parallel:
```bash
python3 -m pytest tests/ -n auto
```

3. Run specific test categories:
```bash
python3 -m pytest tests/test_fractiverse.py  # System tests
python3 -m pytest tests/test_api.py          # API tests
python3 -m pytest tests/test_logging.py      # Logging tests
```

### Test Outputs

Test artifacts are saved in the following directories:
- `test_outputs/logs/` - Test execution logs
- `test_outputs/visualizations/` - Generated visualizations
- `test_outputs/metrics/` - Collected metrics
- `test_outputs/test_summary.json` - Test execution summary

## Development

1. Format code:
```bash
black .
isort .
```

2. Run linting:
```bash
flake8 .
mypy .
```

## Running the Application

Start the FractiVerse server:
```bash
python3 main.py
```

The server will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Metrics: http://localhost:9090/metrics

## Environment Variables

- `FRACTIVERSE_TESTING`: Set to "true" for test mode
- `PORT`: API server port (default: 8000)
- `METRICS_PORT`: Metrics server port (default: 9090)
- `ENVIRONMENT`: "development" or "production"
- `ALLOWED_ORIGINS`: Comma-separated CORS origins

# FractiVerse 1.0 Python Extensions

## 📌 Extension Operators
This module includes:
- ✅ 3D Cognitive Vector Operators → Assigns spatial coordinates to concepts
- ✅ Unipixel Recursive Operators → Implements fractal memory persistence and PEFF logic
- ✅ FractiChain Persistence Operators → Ensures decentralized, evolving AI cognition
- ✅ FractiNet Communication Operators → Allows AI-to-AI distributed learning

## 🚀 Installation

pip install fractiverse


## 🔄 Development Branches

### operator-integration
- Purpose: Integrate new FractiVerse 1.0 Python Extension Operators
- Status: In Progress
- Documentation: [Operator Integration](docs/OPERATOR_INTEGRATION.md)
- Test Suite: [Integration Tests](tests/test_integration.py)

### Testing
eof
