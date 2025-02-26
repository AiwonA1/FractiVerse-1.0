"""FractiVerse 1.0 Main Application Tests"""
import os
import pytest
import json
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import time

from main import (
    app, orchestrator, config,
    TEST_OUTPUT_DIR, TEST_LOG_DIR, TEST_VIZ_DIR, TEST_METRICS_DIR
)
from core.fractiverse_config import load_config
from core.logging_config import setup_logging

from fractiverse.core.fractiverse_orchestrator import FractiVerseOrchestrator
from fractiverse.core.unipixel_core import UnipixelCore
from fractiverse.core.reality_system import RealitySystem
from fractiverse.core.peff_system import PeffSystem
from fractiverse.core.cognitive_engine import CognitiveEngine

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment"""
    # Set testing environment variable
    os.environ["FRACTIVERSE_TESTING"] = "true"
    
    # Ensure test directories exist
    for dir_path in [TEST_OUTPUT_DIR, TEST_LOG_DIR, TEST_VIZ_DIR, TEST_METRICS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Setup test logging
    setup_logging(
        app_name="test_fractiverse",
        output_dir=TEST_LOG_DIR,
        level="DEBUG",
        capture_traces=True
    )
    
    yield
    
    # Save test summary
    save_test_summary()
    
    # Don't cleanup test outputs - they should be available for inspection

def save_test_summary():
    """Save test execution summary"""
    try:
        summary_file = TEST_OUTPUT_DIR / "test_summary.json"
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "test_files": [
                str(f.relative_to(TEST_OUTPUT_DIR))
                for f in TEST_OUTPUT_DIR.rglob("*")
                if f.is_file()
            ],
            "log_files": [
                str(f.relative_to(TEST_LOG_DIR))
                for f in TEST_LOG_DIR.glob("*.log")
            ],
            "visualization_files": [
                str(f.relative_to(TEST_VIZ_DIR))
                for f in TEST_VIZ_DIR.glob("*.json")
            ],
            "metrics_files": [
                str(f.relative_to(TEST_METRICS_DIR))
                for f in TEST_METRICS_DIR.glob("*.json")
            ]
        }
        
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)
            
    except Exception as e:
        print(f"Failed to save test summary: {e}")

@pytest.fixture
def mock_orchestrator():
    mock = MagicMock()
    # Make async methods return awaitable futures
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    mock.process_input = AsyncMock()
    
    # Set up JSON-serializable return values
    mock.get_metrics.return_value = {
        "system_metrics": {
            "cpu_usage": 0.5,
            "memory_usage": 1024,
            "uptime": 3600
        },
        "component_metrics": {
            "unipixel": {"active": True, "operations": 100},
            "reality": {"active": True, "field_strength": 0.8},
            "peff": {"active": True, "coordinates": []},
            "cognition": {"active": True, "level": 0.7}
        }
    }
    
    mock.get_health.return_value = {
        "status": "healthy",
        "components": {
            "unipixel": True,
            "reality": True,
            "peff": True,
            "cognition": True
        },
        "timestamp": datetime.now().isoformat()
    }
    
    mock.get_state.return_value = {
        "cognitive_state": {
            "last_input": {},
            "peff_state": {
                "coordinates": [],
                "field_state": {}
            }
        }
    }
    
    return mock

def test_config_loading():
    """Test configuration loading"""
    assert config is not None
    assert "version" in config
    assert "components" in config
    assert "monitoring" in config

@pytest.mark.asyncio
async def test_startup(mock_orchestrator):
    """Test system startup"""
    app.state.orchestrator = mock_orchestrator
    
    with TestClient(app) as client:
        response = client.get("/startup")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        mock_orchestrator.start.assert_called_once()

@pytest.mark.asyncio
async def test_shutdown(mock_orchestrator):
    """Test system shutdown"""
    app.state.orchestrator = mock_orchestrator
    
    with TestClient(app) as client:
        response = client.get("/shutdown")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        mock_orchestrator.stop.assert_called_once()

@pytest.mark.asyncio
async def test_health_check(mock_orchestrator):
    """Test health check endpoint"""
    app.state.orchestrator = mock_orchestrator
    
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert all(component in data["components"] for component in ["unipixel", "reality", "peff", "cognition"])

@pytest.mark.asyncio
async def test_metrics_endpoint(mock_orchestrator):
    """Test metrics endpoint"""
    app.state.orchestrator = mock_orchestrator
    
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "system_metrics" in data
        assert "component_metrics" in data
        assert all(metric in data["system_metrics"] for metric in ["cpu_usage", "memory_usage", "uptime"])

@pytest.mark.asyncio
async def test_error_handling(mock_orchestrator):
    """Test error handling"""
    app.state.orchestrator = mock_orchestrator
    mock_orchestrator.process_input.side_effect = ValueError("Invalid input")
    
    with TestClient(app) as client:
        response = client.post("/process", json={"invalid": "data"})
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "Invalid input" in data["message"]

@pytest.mark.asyncio
async def test_process_input_valid(mock_orchestrator):
    """Test processing valid input"""
    test_input = {
        "command": "test",
        "parameters": {"key": "value"}
    }
    
    response = client.post("/process", json=test_input)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "command_id" in data
    assert "result" in data
    assert "cognitive_level" in data
    assert "processing_time" in data

def test_process_input_invalid():
    """Test processing invalid input"""
    response = client.post("/process", json={})
    assert response.status_code == 400
    assert "detail" in response.json()

@pytest.mark.asyncio
async def test_component_integration(mock_orchestrator):
    """Test component integration flow"""
    test_input = {"command": "test_integration"}
    
    # Process input
    response = client.post("/process", json=test_input)
    assert response.status_code == 200
    
    # Verify component interactions
    mock_orchestrator.return_value.process_input.assert_called_once_with(test_input)
    
    # Check metrics
    metrics = client.get("/metrics").json()
    assert metrics["requests_total"] > 0
    assert metrics["memory_usage"] == 100
    assert metrics["network_peers"] == 5

@pytest.mark.asyncio
async def test_concurrent_processing(mock_orchestrator):
    """Test concurrent request processing"""
    async def make_request():
        return client.post("/process", json={"command": "concurrent_test"})
    
    # Make concurrent requests
    tasks = [make_request() for _ in range(5)]
    responses = await asyncio.gather(*tasks)
    
    # Verify all requests succeeded
    assert all(r.status_code == 200 for r in responses)
    assert mock_orchestrator.return_value.process_input.call_count == 5

def test_visualization_output(mock_orchestrator):
    """Test visualization output"""
    # Process request to trigger visualization
    response = client.post("/process", json={"command": "viz_test"})
    assert response.status_code == 200
    
    # Check health visualization
    response = client.get("/health")
    assert response.status_code == 200
    
    # Verify visualization files were created
    viz_dir = Path("visualizations")
    viz_files = list(viz_dir.glob("*.json"))
    assert len(viz_files) > 0
    
    # Verify visualization content
    with open(viz_files[0], "r") as f:
        data = json.load(f)
        assert "name" in data
        assert "type" in data
        assert "data" in data

def verify_test_outputs():
    """Verify test outputs were generated"""
    # Check log files
    log_files = list(TEST_LOG_DIR.glob("*.log"))
    assert len(log_files) > 0, "No log files generated"
    
    # Check visualization files
    viz_files = list(TEST_VIZ_DIR.glob("*.json"))
    assert len(viz_files) > 0, "No visualization files generated"
    
    # Check metrics files
    metrics_files = list(TEST_METRICS_DIR.glob("*.json"))
    assert len(metrics_files) > 0, "No metrics files generated"
    
    # Verify log content
    with open(log_files[0], "r") as f:
        content = f.read()
        assert "FractiVerse" in content
        assert "test_fractiverse" in content
    
    # Verify visualization content
    with open(viz_files[0], "r") as f:
        data = json.load(f)
        assert isinstance(data, dict)
        assert "data" in data
    
    # Verify metrics content
    with open(metrics_files[0], "r") as f:
        data = json.load(f)
        assert isinstance(data, dict)
        assert any(key in data for key in [
            "requests_total",
            "cognitive_level",
            "memory_usage",
            "network_peers"
        ])

@pytest.mark.asyncio
async def test_full_system_flow(mock_orchestrator):
    """Test full system flow with output verification"""
    # Start system
    await app.router.startup()
    
    # Process multiple requests
    test_inputs = [
        {"command": "test1", "data": "value1"},
        {"command": "test2", "data": "value2"},
        {"command": "test3", "data": "value3"}
    ]
    
    for test_input in test_inputs:
        response = client.post("/process", json=test_input)
        assert response.status_code == 200
        
        # Check health after each request
        health = client.get("/health").json()
        assert health["status"] == "healthy"
        
        # Check metrics after each request
        metrics = client.get("/metrics").json()
        assert metrics["requests_total"] > 0
    
    # Stop system
    await app.router.shutdown()
    
    # Verify outputs
    verify_test_outputs()

@pytest.fixture
async def orchestrator():
    """Create a real instance of FractiVerseOrchestrator for testing."""
    from fractiverse.core.fractiverse_orchestrator import FractiVerseOrchestrator
    
    orchestrator = FractiVerseOrchestrator()
    await orchestrator.start()
    
    yield orchestrator
    
    # Clean up
    await orchestrator.stop()

@pytest.mark.asyncio
async def test_orchestrator_startup(orchestrator):
    """Test orchestrator startup."""
    assert orchestrator is not None
    
    # Check component initialization
    for component in orchestrator.components.values():
        assert component.active is True
    
    # Verify orchestrator state
    assert orchestrator.get_health()["status"] == "healthy"
    metrics = orchestrator.get_metrics()
    assert "system_metrics" in metrics
    assert "component_metrics" in metrics

@pytest.mark.asyncio
async def test_system_shutdown(orchestrator):
    """Test system shutdown process."""
    # Stop the orchestrator
    await orchestrator.stop()
    
    # Check that all components are inactive
    for component in orchestrator.components.values():
        assert not component.active
    
    # Reinitialize for other tests
    await orchestrator.start()

@pytest.mark.asyncio
async def test_process_input(orchestrator):
    """Test input processing."""
    # Start the orchestrator
    await orchestrator.start()
    
    # Test input
    test_input = {
        "coordinates": [
            {
                "position": [1, 1, 1],
                "value": 0.5
            }
        ]
    }
    
    # Process input
    result = await orchestrator.process_input(test_input)
    
    # Check result
    assert result["status"] == "success"
    assert "command_id" in result
    assert "cognitive_state" in result
    assert "reality_state" in result
    assert "peff_state" in result

@pytest.mark.asyncio
async def test_component_interaction(orchestrator):
    """Test component interaction."""
    # Start the orchestrator
    await orchestrator.start()
    
    # Test input that requires all components
    test_input = {
        "coordinates": [
            {
                "position": [1, 1, 1],
                "value": 0.5
            },
            {
                "position": [2, 2, 2],
                "value": 0.7
            }
        ]
    }
    
    # Process input
    result = await orchestrator.process_input(test_input)
    
    # Check result shows interaction between components
    assert result["status"] == "success"
    assert result["cognitive_state"]  # Should have cognitive processing results
    assert result["reality_state"]    # Should have reality processing results
    assert result["peff_state"]       # Should have PEFF processing results
    
    # Check that unipixel space was updated
    unipixel = orchestrator.components["unipixel"]
    assert unipixel.get_point(1, 1, 1) is not None
    assert unipixel.get_point(2, 2, 2) is not None

def test_logging_output():
    """Test logging output"""
    # Wait briefly for logs to be written
    time.sleep(0.1)
    
    # Get the test outputs directory
    test_outputs = Path("test_outputs")
    current_run = max((d for d in test_outputs.iterdir() if d.is_dir()), key=lambda x: x.stat().st_mtime)
    log_dir = current_run / "logs"
    
    # Get the most recent log file
    log_files = list(log_dir.glob("test_fractiverse_*.log"))
    assert len(log_files) > 0, "No log files found"
    
    log_file = max(log_files, key=lambda x: x.stat().st_mtime)
    
    # Read the log file
    with open(log_file, "r") as f:
        content = f.read()
    
    # Check for expected log entries
    assert "FractiVerse" in content, "Missing FractiVerse in log output"
    assert "Starting" in content, "Missing startup message"
    assert "INFO" in content, "Missing INFO level messages"
    assert "DEBUG" in content, "Missing DEBUG level messages"
    
    # Check structured logging format
    assert '"event":' in content, "Missing event field"
    assert '"level":' in content, "Missing level field"
    assert '"timestamp":' in content, "Missing timestamp field"
    
    # Verify test-specific entries
    assert "test_fractiverse" in content, "Missing test module name"
    assert "Starting test suite" in content, "Missing test suite start message"

if __name__ == "__main__":
    pytest.main(["-v", "--capture=no"])
