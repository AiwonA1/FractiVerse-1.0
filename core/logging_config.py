import os
import logging
import structlog
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Dict

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Create visualization directory if it doesn't exist
VIZ_DIR = Path("visualizations")
VIZ_DIR.mkdir(exist_ok=True)

def setup_logging(app_name: str = "fractiverse", output_dir: Path = None, level: str = "INFO", capture_traces: bool = False) -> None:
    """Configure structured logging with file rotation and console output
    
    Args:
        app_name: Name of the application for log file naming
        output_dir: Optional custom output directory for logs
        level: Logging level (default: INFO)
        capture_traces: Whether to capture stack traces (default: False)
    """
    
    # Use provided output directory or default LOG_DIR
    log_dir = output_dir if output_dir else LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{app_name}_{timestamp}.log"
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Configure structured logging
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ]
    
    if capture_traces:
        processors.insert(-2, structlog.processors.StackInfoRenderer())
    
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def log_metric(metric_name: str, value: Any, metadata: Dict[str, Any] = None) -> None:
    """Log metrics with optional metadata for visualization"""
    timestamp = datetime.now().isoformat()
    metric_data = {
        "timestamp": timestamp,
        "metric": metric_name,
        "value": value,
        "metadata": metadata or {}
    }
    
    # Save to metrics log file
    metrics_file = LOG_DIR / "metrics.jsonl"
    with open(metrics_file, "a") as f:
        f.write(json.dumps(metric_data) + "\n")

def save_visualization(viz_name: str, data: Dict[str, Any], viz_type: str = "line") -> str:
    """Save visualization data for later rendering"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_file = VIZ_DIR / f"{viz_name}_{timestamp}.json"
    
    viz_data = {
        "name": viz_name,
        "type": viz_type,
        "timestamp": timestamp,
        "data": data
    }
    
    with open(viz_file, "w") as f:
        json.dump(viz_data, f, indent=2)
    
    return str(viz_file)

class SafeLogger:
    """Thread-safe logger with failure handling"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self.fallback_log = LOG_DIR / "fallback.log"
    
    def _safe_log(self, level: str, message: str, **kwargs):
        """Log with fallback on failure"""
        try:
            log_func = getattr(self.logger, level)
            log_func(message, **kwargs)
        except Exception as e:
            # Fallback to simple file logging
            timestamp = datetime.now().isoformat()
            with open(self.fallback_log, "a") as f:
                f.write(f"{timestamp} [{level.upper()}] {message} {json.dumps(kwargs)}\n")
                f.write(f"{timestamp} [ERROR] Logging failed: {str(e)}\n")
    
    def info(self, message: str, **kwargs):
        self._safe_log("info", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._safe_log("error", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._safe_log("warning", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self._safe_log("debug", message, **kwargs) 