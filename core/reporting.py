"""HTML report generation for FractiVerse"""
import os
import json
import html
import webbrowser
import shutil
from pathlib import Path
from datetime import datetime, timezone
import structlog
from typing import Dict, Any

logger = structlog.get_logger("fractiverse_reporting")

def generate_html_report(test_output_dir: Path, test_results: Dict[str, Any], run_id: str):
    """Generate a comprehensive HTML test report with interactive features"""
    try:
        report_dir = test_output_dir / "report"
        report_dir.mkdir(exist_ok=True)
        
        # Copy visualizations to report directory
        viz_report_dir = report_dir / "visualizations"
        viz_report_dir.mkdir(exist_ok=True)
        
        viz_dir = test_output_dir / "visualizations"
        for viz_file in viz_dir.rglob("*.png"):
            shutil.copy2(viz_file, viz_report_dir / viz_file.name)
        
        # Calculate test statistics
        total_tests = sum(sum(results.values()) for results in test_results.values())
        total_passed = sum(results["passed"] for results in test_results.values())
        total_failed = sum(results["failed"] for results in test_results.values())
        total_skipped = sum(results["skipped"] for results in test_results.values())
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Read log files
        log_entries = []
        log_dir = test_output_dir / "logs"
        for log_file in log_dir.glob("*.log"):
            with open(log_file) as f:
                log_entries.extend([line.strip() for line in f if line.strip()])
        
        # Generate HTML content
        html_content = _generate_html_content(
            run_id=run_id,
            test_results=test_results,
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
            total_skipped=total_skipped,
            pass_rate=pass_rate,
            log_entries=log_entries,
            viz_report_dir=viz_report_dir
        )
        
        # Write HTML report
        report_file = report_dir / "index.html"
        with open(report_file, "w") as f:
            f.write(html_content)
            
        print("\n✅ HTML test report generated:", report_file)
        logger.info("HTML test report generated", report_file=str(report_file))
        
        # Open report in browser if not in CI environment
        if not os.environ.get("CI"):
            webbrowser.open(f"file://{report_file}")
        
    except Exception as e:
        print(f"\n❌ Failed to generate HTML report: {str(e)}")
        logger.error("Failed to generate HTML report", error=str(e))

def _generate_html_content(run_id: str, test_results: Dict[str, Any], total_tests: int,
                         total_passed: int, total_failed: int, total_skipped: int,
                         pass_rate: float, log_entries: list, viz_report_dir: Path) -> str:
    """Generate the HTML content for the test report"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FractiVerse Test Report - {run_id}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdn.datatables.net/1.11.5/css/dataTables.bootstrap5.min.css" rel="stylesheet">
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.11.5/js/dataTables.bootstrap5.min.js"></script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            .test-summary {{ padding: 20px; }}
            .visualization {{ margin: 20px 0; }}
            .log-viewer {{ 
                max-height: 400px; 
                overflow-y: auto; 
                background: #f8f9fa; 
                padding: 10px;
                font-family: monospace;
            }}
            .status-badge {{
                font-size: 1.2em;
                padding: 5px 10px;
            }}
            .metric-card {{
                margin: 10px;
                min-width: 200px;
            }}
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container-fluid">
                <a class="navbar-brand" href="#">FractiVerse Test Report</a>
                <span class="navbar-text">Run ID: {run_id}</span>
            </div>
        </nav>
        
        <div class="container-fluid mt-4">
            <div class="row">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h4 class="mb-0">Test Summary</h4>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-3">
                                    <div class="card metric-card">
                                        <div class="card-body text-center">
                                            <h5>Total Tests</h5>
                                            <h2>{total_tests}</h2>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="card metric-card bg-success text-white">
                                        <div class="card-body text-center">
                                            <h5>Passed</h5>
                                            <h2>{total_passed}</h2>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="card metric-card bg-danger text-white">
                                        <div class="card-body text-center">
                                            <h5>Failed</h5>
                                            <h2>{total_failed}</h2>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="card metric-card bg-warning">
                                        <div class="card-body text-center">
                                            <h5>Skipped</h5>
                                            <h2>{total_skipped}</h2>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row mt-4">
                                <div class="col-12">
                                    <div class="progress" style="height: 30px;">
                                        <div class="progress-bar bg-success" 
                                             role="progressbar" 
                                             style="width: {pass_rate}%"
                                             aria-valuenow="{pass_rate}" 
                                             aria-valuemin="0" 
                                             aria-valuemax="100">
                                            {pass_rate:.1f}% Pass Rate
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h4 class="mb-0">Module Results</h4>
                        </div>
                        <div class="card-body">
                            <table class="table table-striped" id="moduleTable">
                                <thead>
                                    <tr>
                                        <th>Module</th>
                                        <th>Passed</th>
                                        <th>Failed</th>
                                        <th>Skipped</th>
                                        <th>Pass Rate</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {''.join(
                                        f"<tr>"
                                        f"<td>{module}</td>"
                                        f"<td class='text-success'>{results['passed']}</td>"
                                        f"<td class='text-danger'>{results['failed']}</td>"
                                        f"<td class='text-warning'>{results['skipped']}</td>"
                                        f"<td><div class='progress'>"
                                        f"<div class='progress-bar bg-success' role='progressbar' "
                                        f"style='width: {(results['passed'] / sum(results.values()) * 100) if sum(results.values()) > 0 else 0}%'>"
                                        f"{(results['passed'] / sum(results.values()) * 100) if sum(results.values()) > 0 else 0:.1f}%"
                                        f"</div></div></td></tr>"
                                        for module, results in test_results.items()
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h4 class="mb-0">Test Visualizations</h4>
                        </div>
                        <div class="card-body">
                            <div id="visualizationCarousel" class="carousel slide" data-bs-ride="carousel">
                                <div class="carousel-inner">
                                    {''.join(
                                        f"<div class='carousel-item{' active' if i == 0 else ''}'>"
                                        f"<img src='visualizations/{viz_file.name}' "
                                        f"class='d-block w-100' alt='{viz_file.stem}'>"
                                        f"<div class='carousel-caption d-none d-md-block bg-dark bg-opacity-75'>"
                                        f"<h5>{viz_file.stem}</h5>"
                                        f"</div></div>"
                                        for i, viz_file in enumerate(sorted(viz_report_dir.glob('*.png')))
                                    )}
                                </div>
                                <button class="carousel-control-prev" type="button" 
                                        data-bs-target="#visualizationCarousel" data-bs-slide="prev">
                                    <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                                    <span class="visually-hidden">Previous</span>
                                </button>
                                <button class="carousel-control-next" type="button" 
                                        data-bs-target="#visualizationCarousel" data-bs-slide="next">
                                    <span class="carousel-control-next-icon" aria-hidden="true"></span>
                                    <span class="visually-hidden">Next</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h4 class="mb-0">Test Logs</h4>
                        </div>
                        <div class="card-body">
                            <div class="log-viewer">
                                {'<br>'.join(f'<span class="text-{"success" if "PASS" in entry else "danger" if "FAIL" in entry else "warning" if "SKIP" in entry else "info"}">{html.escape(entry)}</span>' for entry in log_entries)}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            $(document).ready(function() {{
                $('#moduleTable').DataTable({{
                    order: [[4, 'desc']],
                    pageLength: 25
                }});
            }});
        </script>
    </body>
    </html>
    """ 