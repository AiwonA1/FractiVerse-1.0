"""FractiVerse 1.0 Main Application"""
import os
import sys
import asyncio
from typing import Dict, Any
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_client import start_http_server, Counter, Gauge
import json
import structlog
import logging

# Import FractiVerse components
from core.logging_config import setup_logging, log_metric, save_visualization
from core.fractiverse_config import load_config
from fractiverse.core.fractiverse_orchestrator import FractiVerseOrchestrator

# Base directories
BASE_DIR = Path(__file__).parent
TEST_OUTPUT_DIR = BASE_DIR / "test_outputs"
TEST_LOG_DIR = TEST_OUTPUT_DIR / "logs"
TEST_VIZ_DIR = TEST_OUTPUT_DIR / "visualizations"
TEST_METRICS_DIR = TEST_OUTPUT_DIR / "metrics"

# Load environment variables and configuration
load_dotenv()
config = load_config()

# Configuration
PORT = int(os.getenv("PORT", 8000))
METRICS_PORT = int(os.getenv("METRICS_PORT", 9090))
ENV = os.getenv("ENVIRONMENT", "development")
DEBUG = ENV == "development"
IS_TESTING = os.getenv("FRACTIVERSE_TESTING", "false").lower() == "true"

# Create test output directories
for dir_path in [TEST_OUTPUT_DIR, TEST_LOG_DIR, TEST_VIZ_DIR, TEST_METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize logging
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
    
    # Initialize visualization config if not present
    if "visualization" not in config:
        config["visualization"] = {
            "enabled": True,
            "output_dir": str(TEST_VIZ_DIR),
            "format": "png",
            "dpi": 300,
            "style": "seaborn",
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
    else:
        config["visualization"]["output_dir"] = str(TEST_VIZ_DIR)

# Metrics
REQUESTS_TOTAL = Counter('fractiverse_requests_total', 'Total requests processed')
COGNITIVE_LEVEL = Gauge('fractiverse_cognitive_level', 'Current cognitive level')
MEMORY_USAGE = Gauge('fractiverse_memory_usage', 'Memory usage')
NETWORK_PEERS = Gauge('fractiverse_network_peers', 'Connected network peers')

# Initialize FastAPI App
app = FastAPI(
    title="FractiVerse API",
    description="FractiVerse 1.0 - Fractal Intelligence System",
    version="1.0.0",
    debug=DEBUG
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if DEBUG else [os.getenv("ALLOWED_ORIGINS", "").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize FractiVerse Orchestrator
orchestrator = FractiVerseOrchestrator()

@app.on_event("startup")
async def startup_event():
    """Initialize FractiVerse system on startup"""
    try:
        logger.info("Starting FractiVerse 1.0", environment=ENV, testing=IS_TESTING)
        
        # Start metrics server with test configuration
        if config["monitoring"]["prometheus"]:
            metrics_output = TEST_METRICS_DIR if IS_TESTING else None
            start_http_server(METRICS_PORT, dir=metrics_output)
            logger.info("Metrics server started", port=METRICS_PORT, output_dir=metrics_output)
        
        # Start orchestrator
        success = await orchestrator.start()
        if not success:
            logger.error("Failed to start FractiVerse orchestrator")
            raise RuntimeError("Failed to start FractiVerse system")
        
        logger.info("FractiVerse 1.0 started successfully")
        
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        # Save test artifacts before shutdown
        save_test_artifacts()
        
        await orchestrator.stop()
        logger.info("FractiVerse system stopped successfully")
    except Exception as e:
        logger.error("Shutdown error", error=str(e))

@app.post("/process")
async def process_input(request: Dict[str, Any], background_tasks: BackgroundTasks):
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
async def process_command(request: Dict[str, Any], background_tasks: BackgroundTasks):
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

def save_test_artifacts():
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
            
        # Save component state
        state_file = TEST_VIZ_DIR / "final_state.json"
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "components": {
                name: component.status()
                for name, component in orchestrator.components.items()
            }
        }
        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2)
            
        logger.info("Test artifacts saved successfully",
                   metrics_file=str(metrics_file),
                   state_file=str(state_file))
                   
    except Exception as e:
        logger.error("Failed to save test artifacts", error=str(e))

def start_server():
    """Initialize and start the FractiVerse server"""
    try:
        logger.info("Initializing FractiVerse server")
        
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=PORT,
            log_level="info" if DEBUG else "warning",
            reload=DEBUG
        )
        server = uvicorn.Server(config)
        return server
    
    except Exception as e:
        logger.error("Server initialization failed", error=str(e))
        raise

if __name__ == "__main__":
    try:
        server = start_server()
        server.run()
    except Exception as e:
        logger.error("Application failed to start", error=str(e))
        sys.exit(1)
