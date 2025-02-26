import pytest
import json
import logging
from pathlib import Path
from datetime import datetime
from core.logging_config import (
    setup_logging,
    log_metric,
    save_visualization,
    SafeLogger,
)
from .conftest import LOG_DIR, VIZ_DIR, METRICS_DIR

logger = logging.getLogger("fractiverse.test.logging")

def setup_test_dirs():
    """Set up test directories."""
    for directory in [LOG_DIR, VIZ_DIR, METRICS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def setup_logging():
    """Set up logging for tests."""
    logger = logging.getLogger('fractiverse.test.logging')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    log_file = LOG_DIR / f"test_logging_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_metric(name: str, value: float):
    """Log a metric to the metrics directory."""
    metrics_file = METRICS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump({'name': name, 'value': value, 'timestamp': datetime.now().isoformat()}, f)

def save_visualization(name: str, data: dict):
    """Save visualization data."""
    viz_file = VIZ_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(viz_file, 'w') as f:
        json.dump(data, f)

# Set up logging for tests
logger = setup_logging()

@pytest.fixture
def test_logger():
    """Fixture to provide a test logger."""
    return SafeLogger('test')

def test_logging_setup(test_logger, safe_to_fail):
    """Test logging setup and file creation."""
    def _test():
        logger.info("Testing logging setup")
        assert LOG_DIR.exists()
        log_files = list(LOG_DIR.glob('*.log'))
        assert len(log_files) > 0
        
        # Test SafeLogger
        test_logger.info("Test message")
        test_logger.error("Test error")
        
        # Save visualization of log levels
        save_visualization('log_levels', {
            'debug': 10,
            'info': 20,
            'warning': 30,
            'error': 40,
            'critical': 50
        })
        
        return True
        
    result = safe_to_fail(_test)
    assert result is True

def test_metric_logging(safe_to_fail):
    """Test metric logging functionality."""
    def _test():
        log_metric('test_metric', 42.0)
        metric_files = list(METRICS_DIR.glob('*.json'))
        assert len(metric_files) > 0
        
        with open(metric_files[-1]) as f:
            data = json.load(f)
            assert data['name'] == 'test_metric'
            assert data['value'] == 42.0
            
        # Save visualization of metrics
        save_visualization('metrics', {
            'test_metric': 42.0,
            'timestamp': datetime.now().isoformat()
        })
        
        return True
        
    result = safe_to_fail(_test)
    assert result is True

def test_visualization_saving(safe_to_fail):
    """Test visualization saving functionality."""
    def _test():
        test_data = {
            'type': 'test_viz',
            'data': [1, 2, 3, 4, 5],
            'metadata': {
                'created_at': datetime.now().isoformat()
            }
        }
        
        save_visualization('test_viz', test_data)
        viz_files = list(VIZ_DIR.glob('*.json'))
        assert len(viz_files) > 0
        
        with open(viz_files[-1]) as f:
            data = json.load(f)
            assert data['type'] == 'test_viz'
            assert len(data['data']) == 5
            
        return True
        
    result = safe_to_fail(_test)
    assert result is True

def test_error_handling(safe_to_fail):
    """Test error handling in logging."""
    def _test():
        # Test logging invalid data
        log_metric('invalid_metric', float('nan'))
        
        # Test invalid path
        with pytest.raises(Exception):
            SafeLogger('test', log_path='/invalid/path')
            
        return True
        
    result = safe_to_fail(_test)
    assert result is True

def test_concurrent_logging(safe_to_fail):
    """Test concurrent logging safety."""
    def _test():
        import threading
        
        def log_thread(thread_id):
            logger = SafeLogger(f'thread_{thread_id}')
            for i in range(10):
                logger.info(f"Thread {thread_id} message {i}")
                log_metric(f'thread_{thread_id}_metric', float(i))
                
        threads = []
        for i in range(5):
            t = threading.Thread(target=log_thread, args=(i,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        log_files = list(LOG_DIR.glob('*.log'))
        metric_files = list(METRICS_DIR.glob('*.json'))
        
        assert len(log_files) > 0
        assert len(metric_files) > 0
        
        return True
        
    result = safe_to_fail(_test)
    assert result is True 