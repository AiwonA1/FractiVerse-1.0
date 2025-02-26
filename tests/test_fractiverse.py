"""FractiVerse 1.0 Main Application Tests"""
import os
import pytest
import json
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

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
    """Mock FractiVerse orchestrator"""
    with patch("main.FractiVerseOrchestrator") as mock:
        mock.return_value.start = AsyncMock(return_value=True)
        mock.return_value.stop = AsyncMock()
        mock.return_value.process_input = AsyncMock(return_value={
            "status": "success",
            "command_id": "test_123",
            "result": "test_result",
            "cognitive_level": 1.0,
            "processing_time": 0.1
        })
        mock.return_value.components = {
            "cognition": Mock(status=lambda: "active"),
            "memory": Mock(usage=lambda: 100, status=lambda: "ready"),
            "network": Mock(peer_count=lambda: 5, status=lambda: "connected"),
            "blockchain": Mock(status=lambda: "synced")
        }
        yield mock

def test_config_loading():
    """Test configuration loading"""
    assert config is not None
    assert "version" in config
    assert "components" in config
    assert "monitoring" in config

@pytest.mark.asyncio
async def test_startup(mock_orchestrator):
    """Test application startup"""
    with TestClient(app) as client:
        # Trigger startup event
        await app.router.startup()
        
        mock_orchestrator.return_value.start.assert_called_once()
        assert mock_orchestrator.return_value.start.call_count == 1

@pytest.mark.asyncio
async def test_shutdown(mock_orchestrator):
    """Test application shutdown"""
    with TestClient(app) as client:
        # Trigger shutdown event
        await app.router.shutdown()
        
        mock_orchestrator.return_value.stop.assert_called_once()

def test_health_check(mock_orchestrator):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == config["version"]
    assert "components" in data
    
    # Verify component status
    components = data["components"]
    assert all(k in components for k in ["cognition", "memory", "network", "blockchain"])

def test_metrics_endpoint(mock_orchestrator):
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    
    data = response.json()
    assert all(k in data for k in [
        "cognitive_level",
        "memory_usage",
        "network_peers",
        "requests_total"
    ])

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
async def test_error_handling(mock_orchestrator):
    """Test error handling"""
    # Simulate processing error
    mock_orchestrator.return_value.process_input.side_effect = Exception("Test error")
    
    response = client.post("/process", json={"command": "error_test"})
    assert response.status_code == 500
    assert "detail" in response.json()

def test_logging_output():
    """Test logging output"""
    log_files = list(TEST_LOG_DIR.glob("test_fractiverse_*.log"))
    assert len(log_files) > 0
    
    # Verify log content
    with open(log_files[0], "r") as f:
        content = f.read()
        assert "FractiVerse" in content

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
def orchestrator():
    """Create a test orchestrator."""
    return FractiVerseOrchestrator()

def test_orchestrator_initialization(orchestrator):
    """Test orchestrator initialization."""
    assert orchestrator is not None
    assert isinstance(orchestrator.components["unipixel"], UnipixelCore)
    assert isinstance(orchestrator.components["reality"], RealitySystem)
    assert isinstance(orchestrator.components["peff"], PeffSystem)
    assert isinstance(orchestrator.components["cognition"], CognitiveEngine)

@pytest.mark.asyncio
async def test_orchestrator_startup(orchestrator):
    """Test orchestrator startup."""
    success = await orchestrator.start()
    assert success
    
    # Check that all components are active
    for component in orchestrator.components.values():
        assert component.is_active()

@pytest.mark.asyncio
async def test_orchestrator_shutdown(orchestrator):
    """Test orchestrator shutdown."""
    # Start first
    await orchestrator.start()
    
    # Then stop
    await orchestrator.stop()
    
    # Check that all components are inactive
    for component in orchestrator.components.values():
        assert not component.is_active()

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
async def test_error_handling(orchestrator):
    """Test error handling."""
    # Start the orchestrator
    await orchestrator.start()
    
    # Test invalid input
    test_input = {
        "invalid": "data"
    }
    
    # Process input
    result = await orchestrator.process_input(test_input)
    
    # Check result
    assert result["status"] == "error"
    assert "command_id" in result
    assert "error" in result

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

if __name__ == "__main__":
    pytest.main(["-v", "--capture=no"])
