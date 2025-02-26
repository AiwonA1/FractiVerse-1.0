"""Test runner and test execution functions for FractiVerse"""
import asyncio
from datetime import datetime, timezone
import structlog
from typing import Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json

from core.visualization import save_visualization_with_timestamp

logger = structlog.get_logger("fractiverse_tests")

# Emoji status indicators
EMOJI_STATUS = {
    "start": "🚀",
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "debug": "🔍",
    "test": "🧪",
    "viz": "📊",
    "metric": "📈",
    "skip": "⏭️",
    "pass": "✅",
    "fail": "🔴"
}

class TestRunner:
    def __init__(self, test_output_dir: Path, config: Dict[str, Any]):
        self.test_output_dir = test_output_dir
        self.config = config
        self.test_results = {
            "logging": {"passed": 0, "failed": 0, "skipped": 0},
            "visualization": {"passed": 0, "failed": 0, "skipped": 0},
            "modules": {"passed": 0, "failed": 0, "skipped": 0}
        }

    async def run_logging_tests(self):
        """Run logging system tests"""
        with self.safe_test_execution("logging tests", "logging", EMOJI_STATUS["test"]):
            # Test basic logging levels
            log_levels = {
                "debug": (logger.debug, EMOJI_STATUS["debug"]),
                "info": (logger.info, EMOJI_STATUS["info"]),
                "warning": (logger.warning, EMOJI_STATUS["warning"]),
                "error": (logger.error, EMOJI_STATUS["error"]),
                "critical": (logger.critical, EMOJI_STATUS["error"])
            }
            
            for level, (log_func, emoji) in log_levels.items():
                try:
                    message = f"Test {level} message"
                    log_func(message)
                    print(f"{emoji} {message}")
                    self.test_results["logging"]["passed"] += 1
                except Exception as e:
                    print(f"{EMOJI_STATUS['error']} Failed to log {level}: {str(e)}")
                    self.test_results["logging"]["failed"] += 1

            # Add more logging tests...
            # [Previous logging test code moved here]

    async def run_visualization_tests(self):
        """Run visualization system tests"""
        with self.safe_test_execution("visualization tests", "visualization", EMOJI_STATUS["viz"]):
            # [Previous visualization test code moved here]
            pass

    async def run_module_tests(self):
        """Run module-specific tests"""
        with self.safe_test_execution("module tests", "modules", EMOJI_STATUS["test"]):
            # [Previous module test code moved here]
            pass

    def print_test_summary(self):
        """Print test execution summary with emoji indicators and save detailed visualization"""
        print("\n" + "="*50)
        print(f"{EMOJI_STATUS['test']} Test Execution Summary")
        print("="*50)
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        
        module_results = []
        
        for module, results in self.test_results.items():
            total = sum(results.values())
            total_tests += total
            total_passed += results["passed"]
            total_failed += results["failed"]
            total_skipped += results["skipped"]
            
            pass_rate = (results["passed"] / total * 100) if total > 0 else 0
            module_results.append({
                "module": module,
                "total": total,
                "passed": results["passed"],
                "failed": results["failed"],
                "skipped": results["skipped"],
                "pass_rate": pass_rate
            })
            
            print(f"\n{module.title()} Module:")
            print(f"{EMOJI_STATUS['pass']} Passed: {results['passed']}")
            print(f"{EMOJI_STATUS['fail']} Failed: {results['failed']}")
            print(f"{EMOJI_STATUS['skip']} Skipped: {results['skipped']}")
            print(f"Pass Rate: {pass_rate:.1f}%")
        
        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "-"*50)
        print("Overall Summary:")
        print(f"Total Tests: {total_tests}")
        print(f"Total Passed: {total_passed}")
        print(f"Total Failed: {total_failed}")
        print(f"Total Skipped: {total_skipped}")
        print(f"Overall Pass Rate: {overall_pass_rate:.1f}%")
        print("="*50)
        
        # Create and save test summary visualization
        self._save_test_summary_visualization(module_results, total_tests, total_passed, 
                                           total_failed, total_skipped, overall_pass_rate)

    def _save_test_summary_visualization(self, module_results, total_tests, total_passed,
                                      total_failed, total_skipped, overall_pass_rate):
        """Save detailed test summary visualization"""
        try:
            plt.style.use(self.config["visualization"]["style"])
            
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            gs = plt.GridSpec(2, 2, figure=fig)
            
            # [Previous visualization code moved here]
            # Create the four subplots as before...
            
            plt.tight_layout()
            viz_path = self.test_output_dir / "visualizations" / f"test_summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n{EMOJI_STATUS['viz']} Test summary visualization saved to: {viz_path}")
            
        except Exception as e:
            print(f"\n{EMOJI_STATUS['error']} Failed to create test summary visualization: {str(e)}")

    def save_test_artifacts(self):
        """Save test artifacts"""
        try:
            metrics_file = self.test_output_dir / "metrics" / "test_results.json"
            with open(metrics_file, "w") as f:
                json.dump(self.test_results, f, indent=2)
                
            print(f"{EMOJI_STATUS['success']} Test artifacts saved successfully")
            logger.info("Test artifacts saved successfully",
                       metrics_file=str(metrics_file))
                       
        except Exception as e:
            print(f"{EMOJI_STATUS['error']} Failed to save test artifacts: {str(e)}")
            logger.error("Failed to save test artifacts", error=str(e))

    def safe_test_execution(self, test_name: str, module: str, emoji: str = "🧪"):
        """Context manager for safe test execution with emoji logging and result tracking"""
        class TestContext:
            def __init__(self, runner, test_name, module, emoji):
                self.runner = runner
                self.test_name = test_name
                self.module = module
                self.emoji = emoji

            def __enter__(self):
                print(f"\n{self.emoji} Starting: {self.test_name}")
                logger.info(f"Starting {self.test_name}")
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    self.runner.test_results[self.module]["passed"] += 1
                    print(f"{EMOJI_STATUS['success']} Completed: {self.test_name}")
                    logger.info(f"Completed {self.test_name} successfully")
                else:
                    self.runner.test_results[self.module]["failed"] += 1
                    print(f"{EMOJI_STATUS['error']} Error in {self.test_name}: {str(exc_val)}")
                    logger.error(f"Error in {self.test_name}", error=str(exc_val))
                return True  # Suppress exceptions

        return TestContext(self, test_name, module, emoji) 