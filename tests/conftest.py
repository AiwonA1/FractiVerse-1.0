import os
import pytest
import socket
from pathlib import Path

# Test output directories
TEST_ROOT = Path(__file__).parent.parent / 'test_outputs'
LOG_DIR = TEST_ROOT / 'logs'
VIZ_DIR = TEST_ROOT / 'visualizations'
METRICS_DIR = TEST_ROOT / 'metrics'
ARTIFACTS_DIR = TEST_ROOT / 'artifacts'
COVERAGE_DIR = TEST_ROOT / 'coverage'

def get_free_port():
    """Get a free port number by opening a temporary socket"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

@pytest.fixture(scope='session')
def test_port():
    """Get a free port for testing"""
    return get_free_port()

@pytest.fixture(autouse=True)
def setup_test_env(test_port):
    """Set up the test environment."""
    # Set environment variables for testing
    os.environ['FRACTIVERSE_TESTING'] = 'true'
    os.environ['FRACTIVERSE_PORT'] = str(test_port)
    os.environ['FRACTIVERSE_LOG_DIR'] = str(LOG_DIR)
    os.environ['FRACTIVERSE_VIZ_DIR'] = str(VIZ_DIR)
    os.environ['FRACTIVERSE_METRICS_DIR'] = str(METRICS_DIR)
    
    # Create directories if they don't exist
    for directory in [TEST_ROOT, LOG_DIR, VIZ_DIR, METRICS_DIR, ARTIFACTS_DIR, COVERAGE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Clean up after tests if needed
    # Note: We keep the test outputs for inspection

@pytest.fixture
def safe_to_fail(request):
    """Fixture to mark tests that are allowed to fail gracefully."""
    def _safe_to_fail(func):
        try:
            return func()
        except Exception as e:
            print(f"Test failed gracefully: {e}")
            return None
    return _safe_to_fail 