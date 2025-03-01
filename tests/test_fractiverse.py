"""
FractiVerse System Tests
"""

import pytest
import torch
import asyncio
from core.fractiverse import FractiVerse
from core.cognition.peff import SensoryInput

@pytest.fixture
async def fractiverse():
    """Create FractiVerse instance for testing"""
    system = FractiVerse()
    await system.start()
    return system

@pytest.mark.asyncio
async def test_pattern_processing(fractiverse):
    """Test basic pattern processing"""
    # Create test pattern
    pattern = torch.randn(256, 256).to(fractiverse.device)
    pattern = pattern / torch.norm(pattern)
    
    # Create sensory input
    sensory = SensoryInput(
        visual=pattern,
        auditory=pattern,
        emotional=pattern,
        empathetic=pattern,
        artistic=pattern,
        coherence=0.8
    )
    
    # Process pattern
    result = await fractiverse.process_pattern(pattern, sensory)
    
    assert result is not None
    assert 'pattern_id' in result
    assert result['peff_coherence'] > 0
    assert result['processing_time'] < 1.0

@pytest.mark.asyncio
async def test_memory_integration(fractiverse):
    """Test memory storage and retrieval"""
    # Store pattern
    pattern = torch.randn(256, 256).to(fractiverse.device)
    pattern = pattern / torch.norm(pattern)
    
    result = await fractiverse.process_pattern(pattern)
    pattern_id = result['pattern_id']
    
    # Retrieve pattern
    retrieved = await fractiverse.retrieval.retrieve_pattern(pattern)
    
    assert retrieved is not None
    assert retrieved.similarity > 0.9

@pytest.mark.asyncio
async def test_pattern_evolution(fractiverse):
    """Test pattern evolution"""
    # Create base pattern
    pattern = torch.randn(256, 256).to(fractiverse.device)
    pattern = pattern / torch.norm(pattern)
    
    # Evolve pattern
    evolved = await fractiverse.evolution.evolve_pattern(pattern)
    
    assert evolved is not None
    assert evolved.mutation_score > 0
    assert evolved.novelty_score > 0

@pytest.mark.asyncio
async def test_pattern_emergence(fractiverse):
    """Test pattern emergence"""
    # Create seed patterns
    patterns = [torch.randn(256, 256).to(fractiverse.device) for _ in range(3)]
    patterns = [p / torch.norm(p) for p in patterns]
    
    # Generate emergent pattern
    emerged = await fractiverse.emergence.emerge_pattern(patterns)
    
    assert emerged is not None
    assert emerged.emergence_score > 0
    assert emerged.complexity_level > 0

@pytest.mark.asyncio
async def test_system_stability(fractiverse):
    """Test system stability under load"""
    # Process multiple patterns
    patterns = [torch.randn(256, 256).to(fractiverse.device) for _ in range(10)]
    patterns = [p / torch.norm(p) for p in patterns]
    
    results = []
    for pattern in patterns:
        result = await fractiverse.process_pattern(pattern)
        results.append(result)
        
    # Check metrics
    assert fractiverse.metrics['pattern_count'] >= len(patterns)
    assert fractiverse.metrics['system_coherence'] > 0
    assert all(r['processing_time'] < 1.0 for r in results) 