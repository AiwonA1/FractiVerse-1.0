#!/usr/bin/env python3
"""FractiVerse Test Runner with Enhanced Visualization"""
import os
import sys
import pytest
import logging
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import json
import numpy as np
import psutil

# Create timestamped test output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
TEST_RUN_DIR = Path(__file__).parent.parent / f"test_outputs/run_{timestamp}"
LOG_DIR = TEST_RUN_DIR / "logs"
VIZ_DIR = TEST_RUN_DIR / "visualizations"
METRICS_DIR = TEST_RUN_DIR / "metrics"
ARTIFACTS_DIR = TEST_RUN_DIR / "artifacts"
COVERAGE_DIR = TEST_RUN_DIR / "coverage"

# Create directories
for dir_path in [TEST_RUN_DIR, LOG_DIR, VIZ_DIR, METRICS_DIR, ARTIFACTS_DIR, COVERAGE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "test_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("fractiverse.test_runner")

def create_test_summary_visualization(results: Dict[str, Any]) -> None:
    """Create visual summary of test results"""
    try:
        # Use seaborn style directly
        sns.set_theme(style="whitegrid")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Test results pie chart
        results_data = {
            'Passed': max(0, results.get('passed', 0)),
            'Failed': max(0, results.get('failed', 0)),
            'Skipped': max(0, results.get('skipped', 0)),
            'Error': max(0, results.get('error', 0))
        }
        
        # Only show non-zero values in pie chart
        non_zero_data = {k: v for k, v in results_data.items() if v > 0}
        if non_zero_data:
            colors = {
                'Passed': '#2ecc71',
                'Failed': '#e74c3c',
                'Skipped': '#f1c40f',
                'Error': '#95a5a6'
            }
            wedges, texts, autotexts = ax1.pie(
                [non_zero_data[k] for k in non_zero_data],
                labels=non_zero_data.keys(),
                colors=[colors[k] for k in non_zero_data],
                autopct='%1.1f%%'
            )
        else:
            ax1.text(0.5, 0.5, 'No test results', ha='center', va='center')
        ax1.set_title('Test Results Distribution')
        
        # Coverage metrics
        if results.get('coverage'):
            coverage = results['coverage']
            valid_coverage = {k: v for k, v in coverage.items() if not np.isnan(v)}
            if valid_coverage:
                x_pos = range(len(valid_coverage))
                ax2.bar(x_pos, valid_coverage.values(), color='#2ecc71')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(valid_coverage.keys(), rotation=45)
                ax2.set_title('Code Coverage')
                ax2.set_ylim(0, 100)
            else:
                ax2.text(0.5, 0.5, 'No coverage data', ha='center', va='center')
        else:
            ax2.text(0.5, 0.5, 'No coverage data', ha='center', va='center')
        
        # Error distribution
        if results.get('error_types'):
            error_types = results['error_types']
            valid_errors = {k: v for k, v in error_types.items() if v > 0}
            if valid_errors:
                x_pos = range(len(valid_errors))
                ax3.bar(x_pos, valid_errors.values(), color='#e74c3c')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(valid_errors.keys(), rotation=45)
                ax3.set_title('Error Distribution')
            else:
                ax3.text(0.5, 0.5, 'No errors', ha='center', va='center')
        else:
            ax3.text(0.5, 0.5, 'No error data', ha='center', va='center')
        
        # Test duration distribution
        if results.get('durations'):
            durations = [(name, dur) for name, dur in results['durations'] if not np.isnan(dur)]
            if durations:
                durations = sorted(durations, key=lambda x: x[1], reverse=True)[:10]
                y_pos = range(len(durations))
                ax4.barh(y_pos, [d[1] for d in durations], color='#3498db')
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels([d[0] for d in durations])
                ax4.set_title('Top 10 Slowest Tests')
                ax4.set_xlabel('Duration (seconds)')
            else:
                ax4.text(0.5, 0.5, 'No duration data', ha='center', va='center')
        else:
            ax4.text(0.5, 0.5, 'No duration data', ha='center', va='center')
        
        plt.tight_layout()
        viz_path = VIZ_DIR / "test_summary.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created test summary visualization: {viz_path}")
        
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")

def run_tests(args: argparse.Namespace) -> int:
    """Run test suite with visualization"""
    start_time = datetime.now()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("Starting test suite...")
    
    # Prepare pytest arguments
    pytest_args = [
        '-v',
        '--capture=no',
        '--tb=short',
        '--maxfail=10',
        '--showlocals',
        '--color=yes',
        f'--log-file={LOG_DIR}/pytest.log',
        '--log-file-level=DEBUG',
        '--html=' + str(TEST_RUN_DIR / 'pytest_report.html'),
        '--self-contained-html'
    ]
    
    if args.coverage:
        pytest_args.extend([
            '--cov=fractiverse',
            f'--cov-report=html:{COVERAGE_DIR}',
            '--cov-report=term-missing'
        ])
    
    # Add test path
    pytest_args.append(args.test_path or 'tests/')
    
    # Run tests
    result = pytest.main(pytest_args)
    
    # Collect results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'run_id': run_id,
        'duration': (datetime.now() - start_time).total_seconds(),
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'error': 0,
        'skipped': 0,
        'details': {},
        'metrics': {
            'CPU Usage': psutil.cpu_percent(),
            'Memory Usage': f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB",
            'Python Version': sys.version.split()[0]
        }
    }
    
    # Parse test log
    log_file = LOG_DIR / 'pytest.log'
    error_types = {}
    durations = []
    
    if log_file.exists():
        with open(log_file) as f:
            for line in f:
                if ' PASSED ' in line:
                    summary['passed'] += 1
                    test_name = line.split('[')[0].strip()
                    summary['details'][test_name] = {
                        'status': 'passed',
                        'duration': get_test_duration(line)
                    }
                elif ' FAILED ' in line:
                    summary['failed'] += 1
                    test_name = line.split('[')[0].strip()
                    error_type = line.split(']')[-1].strip().split(':')[0]
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    summary['details'][test_name] = {
                        'status': 'failed',
                        'duration': get_test_duration(line),
                        'error': error_type
                    }
                elif ' ERROR ' in line:
                    summary['error'] += 1
                    test_name = line.split('[')[0].strip()
                    error_type = line.split(']')[-1].strip().split(':')[0]
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    summary['details'][test_name] = {
                        'status': 'error',
                        'duration': get_test_duration(line),
                        'error': error_type
                    }
                elif ' SKIPPED ' in line:
                    summary['skipped'] += 1
                    test_name = line.split('[')[0].strip()
                    summary['details'][test_name] = {
                        'status': 'skipped',
                        'duration': 0
                    }
    
    summary['total_tests'] = (
        summary['passed'] + 
        summary['failed'] + 
        summary['error'] + 
        summary['skipped']
    )
    
    # Generate HTML report
    from core.logging_config import generate_html_report
    generate_html_report(TEST_RUN_DIR, summary, run_id)
    
    # Save summary JSON
    with open(TEST_RUN_DIR / 'test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nTest Summary:")
    logger.info(f"Passed: {summary['passed']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Error: {summary['error']}")
    logger.info(f"Skipped: {summary['skipped']}")
    logger.info(f"Duration: {summary['duration']:.2f}s")
    logger.info(f"\nResults saved to: {TEST_RUN_DIR}")
    
    return result

def get_test_duration(log_line: str) -> float:
    """Extract test duration from log line"""
    try:
        return float(log_line.split('seconds')[-2].split()[-1])
    except:
        return 0.0

def main():
    parser = argparse.ArgumentParser(description='Run FractiVerse test suite')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage')
    parser.add_argument('--test-path', help='Specific test path to run')
    args = parser.parse_args()
    
    sys.exit(run_tests(args))

if __name__ == '__main__':
    main() 