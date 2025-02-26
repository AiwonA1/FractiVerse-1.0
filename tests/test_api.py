import pytest
from fastapi.testclient import TestClient
import asyncio
from unittest.mock import Mock, patch
from main import app, orchestrator
from datetime import datetime
import json

client = TestClient(app)

@pytest.fixture
def mock_orchestrator():
    """Mock the FractiVerse orchestrator"""
    with patch('main.FractiVerseOrchestrator') as mock:
        mock.return_value.cognition.learning_active = True
        mock.return_value.cognition.cognition_level = 1.0
        mock.return_value.memory.get_all.return_value = {}
        mock.return_value.fpu.status.return_value = "ready"
        mock.return_value.decision_engine.status.return_value = "operational"
        yield mock

@pytest.fixture
async def mock_process_command():
    """Mock the process_command method"""
    async def mock_process(*args, **kwargs):
        return {
            "status": "success",
            "response": "Test response",
            "cognition_level": 1.0,
            "memory_size": 0
        }
    with patch.object(orchestrator, 'process_command', mock_process):
        yield

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)

def test_health_check(test_client):
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data
    
    # Check that all components are present
    components = data["components"]
    assert "unipixel" in components
    assert "reality" in components
    assert "peff" in components
    assert "cognition" in components

def test_metrics(test_client):
    """Test the metrics endpoint."""
    response = test_client.get("/metrics")
    assert response.status_code == 200
    
    data = response.json()
    assert "requests" in data
    assert "cognitive_level" in data
    assert "memory_usage" in data
    assert "network_peers" in data

def test_process_input(test_client):
    """Test the process input endpoint."""
    test_input = {
        "coordinates": [
            {
                "position": [1, 1, 1],
                "value": 0.5
            }
        ]
    }
    
    response = test_client.post("/process", json=test_input)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "command_id" in data
    assert "cognitive_state" in data
    assert "reality_state" in data
    assert "peff_state" in data

def test_invalid_input(test_client):
    """Test handling of invalid input."""
    test_input = {
        "invalid": "data"
    }
    
    response = test_client.post("/process", json=test_input)
    assert response.status_code == 500

@pytest.mark.asyncio
async def test_concurrent_requests(test_client):
    """Test concurrent request processing"""
    async def make_request():
        response = test_client.post("/process", json={"command": "concurrent_test"})
        return response

    # Make concurrent requests
    tasks = [make_request() for _ in range(5)]
    responses = await asyncio.gather(*tasks)

    # Verify all requests succeeded
    assert all(r.status_code == 200 for r in responses)
    assert all("command_id" in r.json() for r in responses)
    
    # Verify unique command IDs
    command_ids = [r.json()["command_id"] for r in responses]
    assert len(set(command_ids)) == len(command_ids)

def test_system_startup():
    """Test system startup process."""
    assert orchestrator is not None
    
    # Check that all components are initialized
    assert "unipixel" in orchestrator.components
    assert "reality" in orchestrator.components
    assert "peff" in orchestrator.components
    assert "cognition" in orchestrator.components
    
    # Check that all components are active
    for component in orchestrator.components.values():
        assert component.active

def test_system_shutdown():
    """Test system shutdown process."""
    # Stop the orchestrator
    orchestrator.stop()
    
    # Check that all components are inactive
    for component in orchestrator.components.values():
        assert not component.active

@pytest.mark.asyncio
async def test_command_endpoint_valid(mock_process_command):
    """Test the command endpoint with valid input"""
    test_command = {"command": "test command"}
    response = client.post("/command", json=test_command)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "response" in data
    assert "cognition_level" in data
    assert "memory_size" in data

def test_command_endpoint_empty():
    """Test the command endpoint with empty input"""
    test_command = {"command": ""}
    response = client.post("/command", json=test_command)
    assert response.status_code == 400
    assert "detail" in response.json()
    assert "Invalid input" in response.json()["detail"]

def test_command_endpoint_missing_field():
    """Test the command endpoint with missing command field"""
    test_command = {}
    response = client.post("/command", json=test_command)
    assert response.status_code == 400
    assert "detail" in response.json()

@pytest.mark.asyncio
async def test_metrics_endpoint(mock_orchestrator):
    """Test the metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "cognition_level" in data
    assert "memory_size" in data
    assert "requests_total" in data

@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test orchestrator initialization"""
    assert orchestrator.components["unipixel"] is not None
    assert orchestrator.components["reality"] is not None
    assert orchestrator.components["peff"] is not None
    assert orchestrator.components["cognition"] is not None

@pytest.mark.asyncio
async def test_command_processing_flow():
    """Test the complete command processing flow"""
    test_command = "test flow command"
    
    # Mock component responses
    with patch.object(orchestrator.cognition, 'process_input') as mock_cognition, \
         patch.object(orchestrator.fpu, 'process') as mock_fpu, \
         patch.object(orchestrator.decision_engine, 'evaluate') as mock_decision, \
         patch.object(orchestrator.harmonizer, 'harmonize') as mock_harmonize:
        
        mock_cognition.return_value = "cognitive_result"
        mock_fpu.return_value = "fpu_result"
        mock_decision.return_value = "decision_result"
        mock_harmonize.return_value = "final_result"
        
        response = await orchestrator.process_command(test_command)
        
        # Verify flow
        mock_cognition.assert_called_once_with(test_command)
        mock_fpu.assert_called_once_with("cognitive_result")
        mock_decision.assert_called_once_with("fpu_result")
        mock_harmonize.assert_called_once_with("decision_result")
        
        assert response["status"] == "success"
        assert response["response"] == "final_result"

@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for tests"""
    # Setup
    yield
    # Teardown - cleanup if needed 