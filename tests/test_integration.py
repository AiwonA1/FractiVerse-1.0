import pytest
import json
from fractiverse.operators import FractiVector, Unipixel, FractiChain, FractiNet
from fractiverse.core import CognitiveEngine, RealitySystem, PeffSystem
from fractiverse.core.unipixel_core import UnipixelCore

@pytest.fixture
def cognitive_engine():
    """Create a test cognitive engine."""
    return CognitiveEngine()

@pytest.fixture
def reality_system():
    """Create a test reality system."""
    return RealitySystem()

@pytest.fixture
def peff_system():
    """Create a test PEFF system."""
    return PeffSystem()

@pytest.fixture
def unipixel_core():
    """Create a test unipixel core."""
    return UnipixelCore()

def test_component_initialization(cognitive_engine, reality_system, peff_system, unipixel_core):
    """Test component initialization."""
    assert cognitive_engine is not None
    assert reality_system is not None
    assert peff_system is not None
    assert unipixel_core is not None
    
    assert not cognitive_engine.active
    assert not reality_system.active
    assert not peff_system.active
    assert not unipixel_core.active

def test_component_startup(cognitive_engine, reality_system, peff_system, unipixel_core):
    """Test component startup."""
    assert cognitive_engine.start()
    assert reality_system.start()
    assert peff_system.start()
    assert unipixel_core.start()
    
    assert cognitive_engine.active
    assert reality_system.active
    assert peff_system.active
    assert unipixel_core.active

def test_component_shutdown(cognitive_engine, reality_system, peff_system, unipixel_core):
    """Test component shutdown."""
    # Start components
    cognitive_engine.start()
    reality_system.start()
    peff_system.start()
    unipixel_core.start()
    
    # Stop components
    assert cognitive_engine.stop()
    assert reality_system.stop()
    assert peff_system.stop()
    assert unipixel_core.stop()
    
    assert not cognitive_engine.active
    assert not reality_system.active
    assert not peff_system.active
    assert not unipixel_core.active

def test_data_flow(cognitive_engine, reality_system, peff_system, unipixel_core):
    """Test data flow between components."""
    # Start components
    cognitive_engine.start()
    reality_system.start()
    peff_system.start()
    unipixel_core.start()
    
    # Test input
    test_input = {
        "coordinates": [
            {
                "position": [1, 1, 1],
                "value": 0.5
            }
        ]
    }
    
    # Process through cognitive engine
    cognitive_result = cognitive_engine.process_input(test_input)
    assert cognitive_result is not None
    assert "cognitive_state" in cognitive_result
    
    # Process through reality system
    reality_result = reality_system.process(cognitive_result)
    assert reality_result is not None
    assert "reality_state" in reality_result
    assert "matrix" in reality_result
    
    # Process through PEFF system
    peff_result = peff_system.process(reality_result)
    assert peff_result is not None
    assert "coordinates" in peff_result
    assert "field_state" in peff_result
    
    # Process through unipixel system
    for coord in peff_result["coordinates"]:
        x, y, z = coord["position"]
        value = coord["value"]
        assert unipixel_core.process_point(x, y, z, value)
        assert unipixel_core.get_point(x, y, z) == value

def test_error_handling(cognitive_engine, reality_system, peff_system, unipixel_core):
    """Test error handling between components."""
    # Start components
    cognitive_engine.start()
    reality_system.start()
    peff_system.start()
    unipixel_core.start()
    
    # Test invalid input
    test_input = {
        "invalid": "data"
    }
    
    # Process through cognitive engine
    cognitive_result = cognitive_engine.process_input(test_input)
    assert cognitive_result["status"] == "error"
    assert "invalid input format" in cognitive_result["message"].lower()
    
    # Test invalid coordinates
    test_input = {
        "coordinates": [
            {
                "position": [-1, -1, -1],  # Invalid coordinates
                "value": 0.5
            }
        ]
    }
    
    # Process through cognitive engine
    cognitive_result = cognitive_engine.process_input(test_input)
    assert cognitive_result["status"] == "error"
    assert "invalid coordinates" in cognitive_result["message"].lower()
    
    # Test boundary conditions
    test_input = {
        "coordinates": [
            {
                "position": [float('inf'), 0, 0],  # Infinite coordinate
                "value": 0.5
            }
        ]
    }
    
    # Process through cognitive engine
    cognitive_result = cognitive_engine.process_input(test_input)
    assert cognitive_result["status"] == "error"
    assert "invalid coordinate value" in cognitive_result["message"].lower()
    
    # Test missing fields
    test_input = {
        "coordinates": [
            {
                "position": [0, 0, 0]
                # Missing value field
            }
        ]
    }
    
    # Process through cognitive engine
    cognitive_result = cognitive_engine.process_input(test_input)
    assert cognitive_result["status"] == "error"
    assert "missing required field" in cognitive_result["message"].lower()
    
    # Test type errors
    test_input = {
        "coordinates": [
            {
                "position": ["0", "0", "0"],  # String coordinates instead of numbers
                "value": "0.5"  # String value instead of number
            }
        ]
    }
    
    # Process through cognitive engine
    cognitive_result = cognitive_engine.process_input(test_input)
    assert cognitive_result["status"] == "error"
    assert "invalid data type" in cognitive_result["message"].lower()

@pytest.fixture
def vector():
    """Create a test vector."""
    return FractiVector("Test Thought")

@pytest.fixture
def pixel():
    """Create a test unipixel."""
    return Unipixel("Test_Pixel")

@pytest.fixture
def chain():
    """Create a test chain."""
    return FractiChain()

@pytest.fixture
def network():
    """Create a test network."""
    return FractiNet()

def test_vector_integration(cognitive_engine, vector):
    """Test 3D Cognitive Vector integration"""
    result = cognitive_engine.process_thought(vector)
    assert result is not None

def test_unipixel_integration(pixel):
    """Test Unipixel recursive processing"""
    pixel = pixel >> "Test Knowledge"
    assert "Test Knowledge" in pixel.knowledge

def test_chain_integration(chain):
    """Test FractiChain persistence"""
    chain = chain >> "Test Memory"
    assert "Test Memory" in chain.chain

def test_network_integration(network, pixel):
    """Test FractiNet distribution"""
    # Add a test node to the network
    test_node = {"name": "Test_Node", "type": "test"}
    network.add_node(test_node)
    
    # Add pixel to the network
    network = network | pixel
    
    # Verify both node and pixel are in network
    assert "Test_Node" in network.network
    assert pixel.id in network.network
    
    # Test network operations
    network.process_node(test_node["name"])
    assert network.is_active(test_node["name"])
    
    # Test pixel integration
    pixel_state = network.get_node_state(pixel.id)
    assert pixel_state is not None
    assert "position" in pixel_state

if __name__ == '__main__':
    pytest.main()
