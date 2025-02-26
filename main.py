"""FractiVerse 1.0 Main Application"""
import os
import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import start_http_server, Counter, Gauge, REGISTRY
import structlog
import logging
import json

# Import FractiVerse components
from core.logging_config import setup_logging, log_metric, save_visualization
from core.fractiverse_config import load_config
from fractiverse.core.fractiverse_orchestrator import FractiVerseOrchestrator
from core.utils import find_free_port, force_exit
from core.server import start_server, run_server, shutdown_server, shutdown_after_timeout
from tests.test_runner import TestRunner
from core.reporting import generate_html_report

# Global state
orchestrator = None
server = None
shutdown_event = asyncio.Event()

# Base directories
BASE_DIR = Path(__file__).parent
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
TEST_OUTPUT_DIR = BASE_DIR / "test_outputs" / RUN_ID
TEST_LOG_DIR = TEST_OUTPUT_DIR / "logs"
TEST_VIZ_DIR = TEST_OUTPUT_DIR / "visualizations"
TEST_METRICS_DIR = TEST_OUTPUT_DIR / "metrics"

# Load environment variables and configuration
load_dotenv()
config = load_config()

# Configuration with dynamic port assignment
BASE_PORT = int(os.getenv("PORT", 8000))
BASE_METRICS_PORT = int(os.getenv("METRICS_PORT", 9090))

PORT = find_free_port(BASE_PORT) or BASE_PORT + 100
METRICS_PORT = find_free_port(BASE_METRICS_PORT) or BASE_METRICS_PORT + 100

ENV = os.getenv("ENVIRONMENT", "development")
DEBUG = ENV == "development"
IS_TESTING = os.getenv("FRACTIVERSE_TESTING", "false").lower() == "true"

# Create test output directories with clear structure
for dir_path in [TEST_OUTPUT_DIR, TEST_LOG_DIR, TEST_VIZ_DIR / "gauge", TEST_VIZ_DIR / "status", TEST_VIZ_DIR / "timeline", TEST_METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize logging with test configuration
logger = structlog.get_logger("fractiverse_main")
setup_logging(
    app_name="fractiverse",
    output_dir=TEST_LOG_DIR if IS_TESTING else None,
    level="DEBUG" if IS_TESTING else "INFO",
    capture_traces=IS_TESTING
)

# Test-specific configuration
if IS_TESTING:
    config["logging"]["output_dir"] = str(TEST_LOG_DIR)
    config["monitoring"]["output_dir"] = str(TEST_METRICS_DIR)
    config["visualization"] = {
        "enabled": True,
        "output_dir": str(TEST_VIZ_DIR),
        "format": "png",
        "dpi": 300,
        "style": "seaborn-v0_8-darkgrid",
        "color_scheme": "husl",
        "interactive": False,
        "auto_save": True,
        "types": {
            "test_summary": True,
            "coverage": True,
            "timeline": True,
            "metrics": True,
            "errors": True
        }
    }

# Metrics
REQUESTS_TOTAL = Counter('fractiverse_requests_total', 'Total requests processed')
COGNITIVE_LEVEL = Gauge('fractiverse_cognitive_level', 'Current cognitive level')
MEMORY_USAGE = Gauge('fractiverse_memory_usage', 'Memory usage')
NETWORK_PEERS = Gauge('fractiverse_network_peers', 'Connected network peers')

# Initialize FastAPI App with lifespan
app = FastAPI(
    title="FractiVerse API",
    description="FractiVerse 1.0 - Fractal Intelligence System",
    version="1.0.0",
    debug=DEBUG
)

@app.post("/process")
async def process_input(request: dict, background_tasks: BackgroundTasks):
    """Process input through the FractiVerse system."""
    try:
        result = await orchestrator.process_input(request)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Processing error", error_msg=str(e))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "command_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            }
        )

@app.post("/command")
async def process_command(request: dict, background_tasks: BackgroundTasks):
    """Process a command through the FractiVerse system."""
    try:
        if not request or "command" not in request or not request["command"]:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": "Missing or empty command",
                    "command_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                }
            )
            
        result = await orchestrator.process_command(request)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Command processing error", error_msg=str(e))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "command_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint with component status"""
    try:
        components = {
            name: component.status()
            for name, component in orchestrator.components.items()
        }
        
        status_data = {
            "status": "healthy",
            "version": config["version"],
            "environment": ENV,
            "components": components
        }
        
        # Save health visualization
        save_visualization("health_check", status_data, "status")
        
        return status_data
    
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/metrics")
async def get_metrics():
    """Get current system metrics"""
    try:
        metrics_data = {
            "cognitive_level": COGNITIVE_LEVEL._value.get(),
            "memory_usage": MEMORY_USAGE._value.get(),
            "network_peers": NETWORK_PEERS._value.get(),
            "requests_total": REQUESTS_TOTAL._value.get(),
            "components": {
                name: component.get_metrics()
                for name, component in orchestrator.components.items()
            }
        }
        
        # Save metrics visualization
        save_visualization("metrics", metrics_data, "gauge")
        
        return metrics_data
    
    except Exception as e:
        logger.error("Metrics collection failed", error=str(e))
        return {"error": str(e)}

def save_test_artifacts(test_runner: TestRunner):
    """Save test artifacts if in testing mode"""
    if not IS_TESTING:
        return
        
    try:
        # Save final metrics state
        metrics_data = {
            "requests_total": REQUESTS_TOTAL._value.get(),
            "cognitive_level": COGNITIVE_LEVEL._value.get(),
            "memory_usage": MEMORY_USAGE._value.get(),
            "network_peers": NETWORK_PEERS._value.get()
        }
        metrics_file = TEST_METRICS_DIR / "final_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)
            
        # Save test results
        results_file = TEST_METRICS_DIR / "test_results.json"
        with open(results_file, "w") as f:
            json.dump(test_runner.test_results, f, indent=2)
            
        # Generate HTML report
        generate_html_report(TEST_OUTPUT_DIR, test_runner.test_results, RUN_ID)
            
        print("✅ Test artifacts saved successfully")
        logger.info("Test artifacts saved successfully",
                   metrics_file=str(metrics_file),
                   results_file=str(results_file))
                   
    except Exception as e:
        print(f"❌ Failed to save test artifacts: {str(e)}")
        logger.error("Failed to save test artifacts", error=str(e))

@app.on_event("startup")
async def startup():
    """Initialize the application on startup"""
    global orchestrator
    
    try:
        print("\n🚀 Starting FractiVerse 1.0")
        logger.info("Starting FractiVerse 1.0", environment=ENV, testing=IS_TESTING)
        
        # Initialize orchestrator
        orchestrator = FractiVerseOrchestrator()
        
        # Start metrics server
        if config["monitoring"]["prometheus"]:
            start_http_server(METRICS_PORT)
            print(f"📈 Metrics server started on port {METRICS_PORT}")
            logger.info("Metrics server started", port=METRICS_PORT)
        
        # Run test suites if in testing mode
        if IS_TESTING:
            print("\n🧪 Running test suites...")
            test_runner = TestRunner(TEST_OUTPUT_DIR, config)
            
            # Run tests
            await test_runner.run_logging_tests()
            await test_runner.run_visualization_tests()
            await test_runner.run_module_tests()
            
            # Print test summary and save artifacts
            test_runner.print_test_summary()
            save_test_artifacts(test_runner)
            
            # Start shutdown timeout for testing mode
            asyncio.create_task(shutdown_after_timeout(3, force_exit))
        
        print("\n✅ FractiVerse 1.0 started successfully")
        logger.info("FractiVerse 1.0 started successfully")
        
    except Exception as e:
        print(f"\n❌ Startup failed: {str(e)}")
        logger.error("Startup failed", error=str(e))
        if not IS_TESTING:
            raise

@app.on_event("shutdown")
async def shutdown():
    """Clean up on application shutdown"""
    if orchestrator:
        await orchestrator.stop()

if __name__ == "__main__":
    try:
        server = start_server(app, PORT, DEBUG, IS_TESTING)
        
        # Run server in asyncio event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if IS_TESTING:
            # Add shutdown timeout for testing mode
            loop.create_task(shutdown_after_timeout(3, force_exit))
        
        # Run server
        loop.run_until_complete(run_server(server, IS_TESTING, force_exit))
        
    except KeyboardInterrupt:
        print("\nℹ️ Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error("Application failed to start", error=str(e))
        sys.exit(1)
    finally:
        if loop.is_running():
            loop.close()
        if IS_TESTING:
            force_exit()  # Ensure clean exit in test mode
