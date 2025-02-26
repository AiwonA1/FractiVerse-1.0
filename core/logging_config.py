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

def generate_html_report(output_dir: Path, test_results: Dict[str, Any], run_id: str) -> None:
    """Generate a comprehensive HTML test report with tabbed sections
    
    Args:
        output_dir: Directory to save the report
        test_results: Dictionary containing test results
        run_id: Unique identifier for this test run
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"test_report_{timestamp}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FractiVerse Test Report - {run_id}</title>
        <style>
            :root {{
                --primary-color: #2ecc71;
                --error-color: #e74c3c;
                --warning-color: #f1c40f;
                --info-color: #3498db;
                --bg-color: #f8f9fa;
                --border-color: #dee2e6;
            }}
            
            body {{ 
                font-family: 'Segoe UI', system-ui, sans-serif; 
                margin: 0;
                padding: 2rem;
                background: var(--bg-color);
                color: #333;
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 2rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            h1, h2, h3 {{ 
                color: #2c3e50;
                margin-bottom: 1rem;
            }}
            
            .header {{
                border-bottom: 2px solid var(--border-color);
                padding-bottom: 1rem;
                margin-bottom: 2rem;
            }}
            
            .summary {{
                background: var(--bg-color);
                padding: 1.5rem;
                border-radius: 8px;
                margin-bottom: 2rem;
            }}
            
            .success {{ color: var(--primary-color); }}
            .failure {{ color: var(--error-color); }}
            .warning {{ color: var(--warning-color); }}
            
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1rem;
                margin-bottom: 2rem;
            }}
            
            .metric-card {{
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                border: 1px solid var(--border-color);
            }}
            
            .tabs {{
                display: flex;
                margin-bottom: 2rem;
                border-bottom: 2px solid var(--border-color);
            }}
            
            .tab {{
                padding: 0.75rem 1.5rem;
                cursor: pointer;
                border: none;
                background: none;
                font-size: 1rem;
                color: #666;
                border-bottom: 2px solid transparent;
                margin-bottom: -2px;
            }}
            
            .tab.active {{
                color: var(--primary-color);
                border-bottom: 2px solid var(--primary-color);
            }}
            
            .tab-content {{
                display: none;
                padding: 1rem 0;
            }}
            
            .tab-content.active {{
                display: block;
            }}
            
            .output-section {{
                background: #2c3e50;
                color: #fff;
                padding: 1rem;
                border-radius: 4px;
                font-family: monospace;
                white-space: pre-wrap;
                margin: 1rem 0;
                max-height: 400px;
                overflow-y: auto;
            }}
            
            .badge {{
                display: inline-block;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                font-size: 0.875rem;
                font-weight: 600;
                margin-left: 0.5rem;
            }}
            
            .badge-success {{ background: var(--primary-color); color: white; }}
            .badge-error {{ background: var(--error-color); color: white; }}
            .badge-warning {{ background: var(--warning-color); color: black; }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
            }}
            
            th, td {{
                padding: 0.75rem;
                border: 1px solid var(--border-color);
                text-align: left;
            }}
            
            th {{
                background: var(--bg-color);
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 FractiVerse Test Report</h1>
                <p>Run ID: {run_id}</p>
                <p>Generated: {datetime.now().isoformat()}</p>
            </div>
            
            <div class="tabs">
                <button class="tab active" onclick="openTab(event, 'summary')">Summary</button>
                <button class="tab" onclick="openTab(event, 'details')">Test Details</button>
                <button class="tab" onclick="openTab(event, 'metrics')">System Metrics</button>
                <button class="tab" onclick="openTab(event, 'outputs')">Test Outputs</button>
            </div>
            
            <div id="summary" class="tab-content active">
                <div class="summary">
                    <h2>Test Summary</h2>
                    <p>Total Tests: {test_results.get('total_tests', 0)}</p>
                    <p class="success">✅ Passed: {test_results.get('passed', 0)}</p>
                    <p class="failure">❌ Failed: {test_results.get('failed', 0)}</p>
                    <p class="warning">⚠️ Skipped: {test_results.get('skipped', 0)}</p>
                    <p>⏱️ Duration: {test_results.get('duration', 0):.2f}s</p>
                </div>
                
                <h3>Quick Stats</h3>
                <div class="metrics">
                    <div class="metric-card">
                        <h4>Pass Rate</h4>
                        <p class="success">{(test_results.get('passed', 0) / max(test_results.get('total_tests', 1), 1) * 100):.1f}%</p>
                    </div>
                    <div class="metric-card">
                        <h4>Average Duration</h4>
                        <p>{test_results.get('duration', 0) / max(test_results.get('total_tests', 1), 1):.3f}s per test</p>
                    </div>
                </div>
            </div>
            
            <div id="details" class="tab-content">
                <h2>Test Details</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Test Name</th>
                            <th>Status</th>
                            <th>Duration</th>
                            <th>Error</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Add test details
    for test_name, result in test_results.get('details', {}).items():
        status = result.get('status', 'unknown')
        status_class = 'success' if status == 'passed' else 'failure'
        badge_class = 'badge-success' if status == 'passed' else 'badge-error'
        
        html_content += f"""
                        <tr>
                            <td>{test_name}</td>
                            <td><span class="badge {badge_class}">{status}</span></td>
                            <td>{result.get('duration', 0):.3f}s</td>
                            <td class="failure">{result.get('error', '')}</td>
                        </tr>
        """
    
    html_content += """
                    </tbody>
                </table>
            </div>
            
            <div id="metrics" class="tab-content">
                <h2>System Metrics</h2>
                <div class="metrics">
    """
    
    # Add system metrics
    for metric_name, value in test_results.get('metrics', {}).items():
        html_content += f"""
                    <div class="metric-card">
                        <h4>{metric_name}</h4>
                        <p>{value}</p>
                    </div>
        """
    
    html_content += """
                </div>
            </div>
            
            <div id="outputs" class="tab-content">
                <h2>Test Outputs</h2>
                <div class="output-section">
    """
    
    # Add log output if available
    log_dir = output_dir / "logs"
    if log_dir.exists():
        latest_log = max(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, default=None)
        if latest_log:
            with open(latest_log) as f:
                log_content = f.read()
                html_content += log_content.replace("<", "&lt;").replace(">", "&gt;")
    
    html_content += """
                </div>
            </div>
        </div>
        
        <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            
            // Hide all tab content
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            
            // Remove active class from all tabs
            tablinks = document.getElementsByClassName("tab");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            
            // Show the selected tab content and add active class
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        </script>
    </body>
    </html>
    """
    
    with open(report_file, "w") as f:
        f.write(html_content)
    
    logger.info("Generated HTML test report", report_file=str(report_file)) 