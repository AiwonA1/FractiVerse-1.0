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
    assert cognitive_result is None
    
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
    assert cognitive_result is not None
    
    # Process through reality system
    reality_result = reality_system.process(cognitive_result)
    assert reality_result is not None
    
    # Process through PEFF system
    peff_result = peff_system.process(reality_result)
    assert peff_result is not None
    
    # Process through unipixel system
    for coord in peff_result["coordinates"]:
        x, y, z = coord["position"]
        value = coord["value"]
        # Should fail for invalid coordinates
        if x < 0 or y < 0 or z < 0:
            assert not unipixel_core.process_point(x, y, z, value)
            assert unipixel_core.get_point(x, y, z) is None

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
    network = network | pixel
    assert "Test_Node" in network.network

if __name__ == '__main__':
    pytest.main()
