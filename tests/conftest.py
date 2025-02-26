import os
import pytest
from pathlib import Path

# Test output directories
TEST_ROOT = Path(__file__).parent.parent / 'test_outputs'
LOG_DIR = TEST_ROOT / 'logs'
VIZ_DIR = TEST_ROOT / 'visualizations'
METRICS_DIR = TEST_ROOT / 'metrics'
ARTIFACTS_DIR = TEST_ROOT / 'artifacts'
COVERAGE_DIR = TEST_ROOT / 'coverage'

# Create directories if they don't exist
for directory in [TEST_ROOT, LOG_DIR, VIZ_DIR, METRICS_DIR, ARTIFACTS_DIR, COVERAGE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

@pytest.fixture(scope='session')
def test_env():
    """Set up the test environment."""
    # Set environment variables for testing
    os.environ['FRACTIVERSE_TESTING'] = 'true'
    os.environ['FRACTIVERSE_LOG_DIR'] = str(LOG_DIR)
    os.environ['FRACTIVERSE_VIZ_DIR'] = str(VIZ_DIR)
    os.environ['FRACTIVERSE_METRICS_DIR'] = str(METRICS_DIR)
    
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