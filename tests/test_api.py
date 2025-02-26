import pytest
from fastapi.testclient import TestClient
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from main import app
from fractiverse.core.fractiverse_orchestrator import FractiVerseOrchestrator
from datetime import datetime
import json

client = TestClient(app)

@pytest.fixture
async def orchestrator():
    """Create a real orchestrator instance for testing"""
    orchestrator = FractiVerseOrchestrator()
    await orchestrator.start()
    yield orchestrator
    await orchestrator.stop()

@pytest.fixture
def mock_orchestrator():
    """Mock the FractiVerse orchestrator"""
    mock = AsyncMock(spec=FractiVerseOrchestrator)
    mock.components = {
        "unipixel": AsyncMock(),
        "reality": AsyncMock(),
        "peff": AsyncMock(),
        "cognition": AsyncMock()
    }
    mock.active = True
    with patch('main.orchestrator', mock):
        yield mock

@pytest.fixture
async def mock_process_command(mock_orchestrator):
    """Mock the process_command method"""
    async def mock_process(*args, **kwargs):
        return {
            "status": "success",
            "response": "Test response",
            "cognition_level": 1.0,
            "memory_size": 100
        }
    mock_orchestrator.process_command = mock_process
    return mock_orchestrator

@pytest.fixture
def test_client(mock_orchestrator):
    """Create a test client for the FastAPI application."""
    return TestClient(app)

@pytest.mark.asyncio
async def test_health_check(test_client, mock_orchestrator):
    """Test the health check endpoint."""
    mock_orchestrator.components = {
        "unipixel": AsyncMock(active=True),
        "reality": AsyncMock(active=True),
        "peff": AsyncMock(active=True),
        "cognition": AsyncMock(active=True)
    }
    
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

@pytest.mark.asyncio
async def test_metrics(test_client, mock_orchestrator):
    """Test the metrics endpoint."""
    mock_orchestrator.get_metrics = AsyncMock(return_value={
        "requests": 100,
        "cognitive_level": 0.8,
        "memory_usage": 0.5,
        "network_peers": 3
    })
    
    response = test_client.get("/metrics")
    assert response.status_code == 200
    
    data = response.json()
    assert "requests" in data
    assert "cognitive_level" in data
    assert "memory_usage" in data
    assert "network_peers" in data

@pytest.mark.asyncio
async def test_process_input(test_client, mock_orchestrator):
    """Test the process input endpoint."""
    mock_orchestrator.process_input = AsyncMock(return_value={
        "status": "success",
        "command_id": "test_123",
        "cognitive_state": {"level": 0.8},
        "reality_state": {"active": True},
        "peff_state": {"efficiency": 0.9}
    })
    
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

@pytest.mark.asyncio
async def test_invalid_input(test_client):
    """Test handling of invalid input."""
    test_input = {
        "invalid": "data"
    }
    
    response = test_client.post("/process", json=test_input)
    assert response.status_code == 500

@pytest.mark.asyncio
async def test_concurrent_requests(test_client, mock_orchestrator):
    """Test concurrent request processing"""
    mock_orchestrator.process_input = AsyncMock(return_value={
        "status": "success",
        "command_id": "test_123",
        "cognitive_state": {"level": 0.8},
        "reality_state": {"active": True},
        "peff_state": {"efficiency": 0.9}
    })
    
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

@pytest.mark.asyncio
async def test_system_startup(mock_orchestrator):
    """Test system startup process."""
    assert mock_orchestrator is not None
    
    # Check that all components are initialized
    assert mock_orchestrator.components is not None
    assert "unipixel" in mock_orchestrator.components
    assert "reality" in mock_orchestrator.components
    assert "peff" in mock_orchestrator.components
    assert "cognition" in mock_orchestrator.components
    
    # Check that all components are active
    for component in mock_orchestrator.components.values():
        component.active = True
        assert component.active

@pytest.mark.asyncio
async def test_system_shutdown(mock_orchestrator):
    """Test system shutdown process."""
    # Stop the orchestrator
    await mock_orchestrator.stop()
    
    # Check that all components are inactive
    for component in mock_orchestrator.components.values():
        component.active = False
        assert not component.active

@pytest.mark.asyncio
async def test_orchestrator_initialization(mock_orchestrator):
    """Test orchestrator initialization"""
    assert mock_orchestrator.components is not None
    assert mock_orchestrator.components["unipixel"] is not None
    assert mock_orchestrator.components["reality"] is not None
    assert mock_orchestrator.components["peff"] is not None
    assert mock_orchestrator.components["cognition"] is not None

@pytest.mark.asyncio
async def test_command_processing_flow(mock_orchestrator):
    """Test the complete command processing flow"""
    test_command = "test flow command"
    
    # Mock component responses
    mock_orchestrator.cognition.process_input = AsyncMock(return_value="cognitive_result")
    mock_orchestrator.fpu = AsyncMock()
    mock_orchestrator.fpu.process = AsyncMock(return_value="fpu_result")
    mock_orchestrator.decision_engine = AsyncMock()
    mock_orchestrator.decision_engine.evaluate = AsyncMock(return_value="decision_result")
    mock_orchestrator.harmonizer = AsyncMock()
    mock_orchestrator.harmonizer.harmonize = AsyncMock(return_value="final_result")
    
    response = await mock_orchestrator.process_command(test_command)
    
    # Verify flow
    mock_orchestrator.cognition.process_input.assert_called_once_with(test_command)
    mock_orchestrator.fpu.process.assert_called_once_with("cognitive_result")
    mock_orchestrator.decision_engine.evaluate.assert_called_once_with("fpu_result")
    mock_orchestrator.harmonizer.harmonize.assert_called_once_with("decision_result")
    
    assert response["status"] == "success"
    assert response["response"] == "final_result"

@pytest.mark.asyncio
async def test_command_endpoint_valid(test_client, mock_process_command):
    """Test the command endpoint with valid input"""
    test_command = {"command": "test command"}
    response = test_client.post("/command", json=test_command)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "response" in data
    assert "cognition_level" in data
    assert "memory_size" in data

def test_command_endpoint_empty(test_client):
    """Test the command endpoint with empty input"""
    test_command = {"command": ""}
    response = test_client.post("/command", json=test_command)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "Missing or empty command" in data["error"]

def test_command_endpoint_missing_field(test_client):
    """Test the command endpoint with missing command field"""
    test_command = {}
    response = test_client.post("/command", json=test_command)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "Missing or empty command" in data["error"]

@pytest.mark.asyncio
async def test_metrics_endpoint(test_client, mock_orchestrator):
    """Test the metrics endpoint"""
    mock_orchestrator.get_metrics = AsyncMock(return_value={
        "cognition_level": 0.8,
        "memory_size": 100,
        "requests_total": 50
    })
    
    response = test_client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "cognition_level" in data
    assert "memory_size" in data
    assert "requests_total" in data

@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for tests"""
    # Setup
    yield
    # Teardown - cleanup if needed 