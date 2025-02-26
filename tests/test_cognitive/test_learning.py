"""Tests for cognitive learning system."""
import pytest
import numpy as np
from fractiverse.cognitive.learning import (
    LearningSystem,
    PatternRecognition,
    AdaptiveMemory
)

@pytest.fixture
async def learning_system():
    """Provide a test instance of the learning system."""
    system = LearningSystem(test_mode=True)
    await system.initialize()
    yield system
    await system.shutdown()

@pytest.fixture
def test_patterns():
    """Generate test patterns for learning."""
    return {
        "simple": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),  # Cross pattern
        "complex": np.random.choice([0, 1], size=(5, 5), p=[0.7, 0.3]),  # Random sparse pattern
        "random": np.random.rand(3, 3)  # Continuous random values
    }

@pytest.mark.asyncio
async def test_pattern_learning(learning_system, test_patterns):
    """Test pattern learning capabilities."""
    # Train on simple pattern
    result = await learning_system.learn_pattern(test_patterns["simple"])
    assert result.success
    assert result.confidence > 0.8
    
    # Verify pattern is stored
    stored_patterns = await learning_system.get_learned_patterns()
    assert len(stored_patterns) > 0
    assert np.any(np.isclose(stored_patterns[0], test_patterns["simple"]))

@pytest.mark.asyncio
async def test_adaptive_memory(learning_system, test_patterns):
    """Test adaptive memory functionality."""
    # Initial memory state
    initial_capacity = await learning_system.get_memory_capacity()
    
    # Learn multiple patterns
    for pattern_name in ["simple", "complex"]:
        await learning_system.learn_pattern(test_patterns[pattern_name])
    
    # Check memory adaptation
    new_capacity = await learning_system.get_memory_capacity()
    assert new_capacity >= initial_capacity
    
    # Verify pattern retention
    memory_usage = await learning_system.get_memory_usage()
    assert 0.0 <= memory_usage <= 1.0

@pytest.mark.asyncio
async def test_learning_rate_adjustment(learning_system):
    """Test learning rate adaptation."""
    initial_rate = learning_system.learning_rate
    
    # Train with simple patterns
    for _ in range(5):
        pattern = np.random.choice([0, 1], size=(3, 3))
        await learning_system.learn_pattern(pattern)
    
    # Check learning rate adaptation
    assert learning_system.learning_rate != initial_rate
    assert 0.0 < learning_system.learning_rate < 1.0

@pytest.mark.asyncio
async def test_knowledge_persistence(learning_system, test_patterns):
    """Test knowledge persistence across sessions."""
    # Learn initial pattern
    await learning_system.learn_pattern(test_patterns["simple"])
    
    # Save state
    state_id = await learning_system.save_state()
    assert state_id is not None
    
    # Clear system
    await learning_system.clear()
    
    # Restore state
    success = await learning_system.load_state(state_id)
    assert success
    
    # Verify pattern recognition still works
    recognition = await learning_system.recognize_pattern(test_patterns["simple"])
    assert recognition.confidence > 0.7

@pytest.mark.parametrize("input_pattern,expected_recognition", [
    ("simple", True),
    ("complex", True),
    ("random", False)
])
@pytest.mark.asyncio
async def test_pattern_recognition(learning_system, test_patterns, input_pattern, expected_recognition):
    """Test pattern recognition with different inputs."""
    # Train system on simple and complex patterns
    await learning_system.learn_pattern(test_patterns["simple"])
    await learning_system.learn_pattern(test_patterns["complex"])
    
    # Test recognition
    result = await learning_system.recognize_pattern(test_patterns[input_pattern])
    
    if expected_recognition:
        assert result.recognized
        assert result.confidence > 0.6
    else:
        assert not result.recognized or result.confidence < 0.3 