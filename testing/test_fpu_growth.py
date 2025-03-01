import asyncio
import pytest
from core.system import FractiSystem
from core.memory_manager import MemoryManager

@pytest.mark.asyncio
async def test_fpu_growth():
    """Test FPU level increases with learning"""
    system = FractiSystem()
    await system.initialize()
    
    initial_fpu = system.cognition.cognitive_metrics['fpu_level']
    
    # Process some test inputs
    test_inputs = [
        "Learning test pattern 1",
        "Processing test data 2",
        "Analyzing test sequence 3"
    ]
    
    for input_text in test_inputs:
        await system.process(input_text)
        await asyncio.sleep(1)
    
    final_fpu = system.cognition.cognitive_metrics['fpu_level']
    
    assert final_fpu > initial_fpu, "FPU level should increase with learning"
    print(f"FPU Growth: {initial_fpu:.4f} -> {final_fpu:.4f}")

async def test_memory_initialization():
    """Test proper memory initialization and FPU starting level"""
    memory = MemoryManager()
    
    # Verify initial FPU level
    assert memory.base_metrics['fpu_level'] == 0.0001, "FPU should start at 0.0001"
    assert memory.cognitive_metrics['fpu_level'] == 0.0001, "Cognitive FPU should match base FPU"
    
    # Verify initial metrics
    assert memory.base_metrics['total_patterns'] == 0
    assert memory.base_metrics['total_connections'] == 0
    assert memory.base_metrics['avg_coherence'] == 0.0

async def test_pattern_storage():
    """Test pattern storage and FPU growth"""
    memory = MemoryManager()
    await memory.initialize()
    
    # Process test input
    test_input = "This is a test pattern"
    result = await memory.process_input(test_input)
    
    # Verify pattern was stored
    assert len(memory.long_term_memory['patterns']) > 0, "Pattern should be stored"
    
    # Verify FPU growth
    assert memory.base_metrics['fpu_level'] > 0.0001, "FPU should grow after pattern storage"
    
    # Verify metrics update
    assert memory.base_metrics['total_patterns'] == 1
    assert memory.cognitive_metrics['pattern_recognition'] > 0

async def test_connection_formation():
    """Test connection formation between patterns"""
    memory = MemoryManager()
    await memory.initialize()
    
    # Store related patterns
    inputs = [
        "Neural networks process data",
        "Data processing in AI",
        "Machine learning processes information"
    ]
    
    for input_text in inputs:
        await memory.process_input(input_text)
    
    # Verify connections formed
    total_connections = sum(len(conns) for conns in memory.long_term_memory['connections'].values())
    assert total_connections > 0, "Connections should be formed between related patterns"

async def test_fpu_growth_constraints():
    """Test FPU growth stays within stage constraints"""
    memory = MemoryManager()
    await memory.initialize()
    
    # Process multiple patterns
    for i in range(10):
        await memory.process_input(f"Test pattern {i}")
        
        # Verify FPU constraints
        stage = memory._get_current_stage()
        constraints = memory.fpu_constraints[stage]
        
        assert constraints['min'] <= memory.base_metrics['fpu_level'] <= constraints['max'], \
            f"FPU level {memory.base_metrics['fpu_level']} should stay within {stage} stage constraints"

async def test_memory_persistence():
    """Test memory persistence and loading"""
    memory = MemoryManager()
    await memory.initialize()
    
    # Store test patterns
    test_patterns = ["Test 1", "Test 2", "Test 3"]
    for pattern in test_patterns:
        await memory.process_input(pattern)
    
    # Save memory
    await memory._persist_long_term_memory()
    
    # Create new instance and load memory
    new_memory = MemoryManager()
    loaded_memory = new_memory._load_long_term_memory()
    
    # Verify patterns were loaded
    assert len(loaded_memory['patterns']) == len(test_patterns), "All patterns should be loaded"

@pytest.mark.asyncio
async def test_realtime_updates():
    """Test real-time metric updates"""
    memory = MemoryManager()
    await memory.initialize()
    
    initial_metrics = memory.cognitive_metrics.copy()
    
    # Process pattern and verify immediate update
    await memory.process_input("Test real-time updates")
    await asyncio.sleep(0.2)  # Wait for update interval
    
    # Verify metrics changed
    assert memory.cognitive_metrics != initial_metrics, "Metrics should update in real-time"
    
    # Verify learning log
    assert len(memory.learning_log) > 0, "Learning activity should be logged"
    assert len(memory.activity_log) > 0, "Activity should be logged"

if __name__ == "__main__":
    asyncio.run(test_memory_initialization())
    asyncio.run(test_pattern_storage())
    asyncio.run(test_connection_formation())
    asyncio.run(test_fpu_growth_constraints())
    asyncio.run(test_memory_persistence())
    asyncio.run(test_realtime_updates()) 