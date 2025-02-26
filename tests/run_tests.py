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

# Create timestamped test output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
TEST_RUN_DIR = Path(__file__).parent.parent / f"run_{timestamp}"
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
    format='\n%(asctime)s [%(levelname)8s] %(message)s\n',
    handlers=[
        logging.FileHandler(LOG_DIR / f"test_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("fractiverse.test_runner")

def create_test_summary_visualization(results: Dict[str, Any]) -> None:
    """Create visual summary of test results with enhanced metrics and error handling"""
    try:
        # Set style with error handling
        try:
            plt.style.use('seaborn')
            sns.set_palette("husl")
        except Exception as e:
            logger.warning(f"Failed to set plot style: {e}, using default style")
            plt.style.use('default')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Test results pie chart (top left)
        ax1 = plt.subplot2grid((2, 3), (0, 0))
        results_data = {
            'Passed': results.get('passed', 0),
            'Failed': results.get('failed', 0),
            'Skipped': results.get('skipped', 0),
            'Error': results.get('error', 0)
        }
        colors = ['#2ecc71', '#e74c3c', '#f1c40f', '#95a5a6']
        wedges, texts, autotexts = ax1.pie(
            results_data.values(), 
            labels=results_data.keys(), 
            colors=colors, 
            autopct='%1.1f%%',
            shadow=True
        )
        ax1.set_title('Test Results Distribution')
        
        # Test duration bar plot (top middle)
        ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
        if 'durations' in results and results['durations']:
            durations = results['durations']
            y_pos = range(len(durations[:10]))
            ax2.barh(y_pos, 
                    [d[1] for d in durations[:10]], 
                    align='center',
                    color='#3498db',
                    alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([d[0] for d in durations[:10]])
            ax2.set_title('Top 10 Slowest Tests')
            ax2.set_xlabel('Duration (seconds)')
        else:
            ax2.text(0.5, 0.5, 'No duration data available', 
                    ha='center', va='center')
            ax2.set_title('Test Durations')
        
        # Coverage metrics (bottom left)
        ax3 = plt.subplot2grid((2, 3), (1, 0))
        if 'coverage' in results and results['coverage']:
            coverage = results['coverage']
            x_pos = range(len(coverage))
            ax3.bar(x_pos, coverage.values(), 
                   color='#2ecc71', 
                   alpha=0.7)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(coverage.keys(), rotation=45, ha='right')
            ax3.set_title('Code Coverage by Module')
            ax3.set_ylim(0, 100)
            for i, v in enumerate(coverage.values()):
                ax3.text(i, v + 1, f'{v}%', ha='center')
        else:
            ax3.text(0.5, 0.5, 'No coverage data available', 
                    ha='center', va='center')
            ax3.set_title('Code Coverage')
        
        # Error distribution (bottom middle)
        ax4 = plt.subplot2grid((2, 3), (1, 1))
        if 'error_types' in results and results['error_types']:
            error_types = results['error_types']
            x_pos = range(len(error_types))
            ax4.bar(x_pos, error_types.values(), 
                   color='#e74c3c',
                   alpha=0.7)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(error_types.keys(), rotation=45, ha='right')
            ax4.set_title('Error Distribution')
        else:
            ax4.text(0.5, 0.5, 'No error data available', 
                    ha='center', va='center')
            ax4.set_title('Error Types')
            
        # Test execution timeline (bottom right)
        ax5 = plt.subplot2grid((2, 3), (1, 2))
        if 'timeline' in results and results['timeline']:
            timeline = results['timeline']
            y_pos = range(len(timeline))
            ax5.scatter([t[1] for t in timeline], y_pos, 
                       c=[t[2] for t in timeline],
                       cmap='RdYlGn',
                       alpha=0.7)
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels([t[0] for t in timeline])
            ax5.set_title('Test Execution Timeline')
            ax5.set_xlabel('Time (s)')
        else:
            ax5.text(0.5, 0.5, 'No timeline data available', 
                    ha='center', va='center')
            ax5.set_title('Test Timeline')
        
        # Adjust layout and save
        plt.tight_layout()
        summary_viz_path = VIZ_DIR / "test_summary.png"
        
        try:
            plt.savefig(summary_viz_path, dpi=300, bbox_inches='tight')
            logger.info(f"Created test summary visualization: {summary_viz_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization: {e}")
        finally:
            plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create test summary visualization: {e}")

def run_tests(args: argparse.Namespace) -> int:
    """Run test suite with enhanced logging and visualization"""
    start_time = datetime.now()
    timeline_data = []
    error_types = {}
    
    try:
        logger.info(f"\n{'='*80}\nStarting FractiVerse Test Suite\n{'='*80}")
        
        # Prepare pytest arguments
        pytest_args = [
            '-v',
            '--capture=no',
            '--tb=short',
            '--strict-markers',
            '--maxfail=10',
            '--showlocals',
            '--color=yes',
            '-p', 'no:warnings',
            f'--log-file={LOG_DIR}/pytest.log',
            '--log-file-level=DEBUG'
        ]
        
        # Add coverage if requested
        if args.coverage:
            pytest_args.extend([
                '--cov=fractiverse',
                f'--cov-report=html:{COVERAGE_DIR}',
                '--cov-report=term-missing'
            ])
        
        # Add test selection if specified
        if args.test_path:
            pytest_args.append(args.test_path)
        else:
            pytest_args.append('tests/')
        
        # Run tests and collect results
        logger.info("Running tests...")
        result = pytest.main(pytest_args)
        
        # Parse test results
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'result_code': result,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'error': 0,
            'duration': (datetime.now() - start_time).total_seconds(),
            'coverage': {},
            'error_types': error_types,
            'timeline': timeline_data
        }
        
        # Parse pytest output for results
        log_file = LOG_DIR / 'pytest.log'
        if log_file.exists():
            with open(log_file, 'r') as f:
                content = f.read()
                summary_data['passed'] = content.count('PASSED')
                summary_data['failed'] = content.count('FAILED')
                summary_data['error'] = content.count('ERROR')
                summary_data['skipped'] = content.count('SKIPPED')
                
                # Extract error types
                for line in content.split('\n'):
                    if 'ERROR' in line or 'FAILED' in line:
                        error_type = line.split(']')[-1].strip().split(':')[0]
                        error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Parse coverage data if available
        coverage_file = COVERAGE_DIR / 'index.html'
        if coverage_file.exists():
            import re
            with open(coverage_file, 'r') as f:
                content = f.read()
                matches = re.findall(r'<td>(\d+)%</td>', content)
                if matches:
                    summary_data['coverage'] = {
                        'total': int(matches[0]),
                        'statements': int(matches[1]) if len(matches) > 1 else 0,
                        'branches': int(matches[2]) if len(matches) > 2 else 0,
                        'functions': int(matches[3]) if len(matches) > 3 else 0
                    }
        
        # Create visualizations
        create_test_summary_visualization(summary_data)
        
        # Save summary
        summary_path = TEST_RUN_DIR / 'test_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Test Summary:")
        logger.info(f"  Passed:  {summary_data['passed']}")
        logger.info(f"  Failed:  {summary_data['failed']}")
        logger.info(f"  Skipped: {summary_data['skipped']}")
        logger.info(f"  Errors:  {summary_data['error']}")
        logger.info(f"  Duration: {summary_data['duration']:.2f}s")
        if summary_data['coverage']:
            logger.info(f"  Coverage: {summary_data['coverage']['total']}%")
        logger.info(f"\nTest outputs available in: {TEST_RUN_DIR}")
        logger.info(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        logger.error(f"Test runner failed: {e}", exc_info=True)
        return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='FractiVerse Test Runner')
    parser.add_argument('--coverage', action='store_true', help='Enable coverage reporting')
    parser.add_argument('--test-path', help='Specific test path to run')
    args = parser.parse_args()
    
    # Set testing environment
    os.environ['FRACTIVERSE_TESTING'] = 'true'
    
    # Run tests
    sys.exit(run_tests(args))

if __name__ == '__main__':
    main() 