"""Test Cognitive Learning and Pattern Formation"""

import asyncio
import pytest
from core.system import FractiSystem

async def input_cognitive_data(system):
    """Input test data and show learning process"""
    
    # Test cognitive inputs
    test_patterns = [
        "What is the relationship between quantum states and neural patterns?",
        "Learning occurs through pattern strengthening and relationship formation",
        "If cognitive coherence increases, then pattern recognition improves",
        "Neural networks exhibit quantum-like interference effects",
        "How does memory integration affect learning efficiency?",
        "Pattern recognition leads to improved cognitive processing because of neural plasticity",
    ]
    
    print("\n🧪 Starting Cognitive Learning Test\n")
    
    for pattern in test_patterns:
        print(f"\n📥 Processing Input: {pattern}")
        
        # Process pattern
        response = await system.process({
            'content': pattern,
            'type': 'cognitive_input'
        })
        
        # Show learning details
        print("\n".join(
            activity['message'] 
            for activity in system.memory.activity_log
        ))
        system.memory.activity_log.clear()
        
        # Let the system process
        await asyncio.sleep(2)
        
        # Show memory state
        print("\n📊 Current Memory State:")
        print(f"├── Patterns: {len(system.memory.patterns)}")
        print(f"├── FPU Level: {system.cognition.cognitive_metrics['fpu_level']*100:.2f}%")
        print(f"└── Integration: {system.memory.memory_metrics['integration_level']*100:.2f}%")
        
        await asyncio.sleep(3)

@pytest.mark.asyncio
async def test_cognitive_learning():
    """Test cognitive learning and pattern formation"""
    
    # Initialize system
    system = FractiSystem()
    await system.initialize()
    
    # Run learning test
    await input_cognitive_data(system)
    
    # Show final memory analysis
    print("\n🔍 Final Memory Analysis:")
    
    # Show pattern relationships
    print("\n📊 Pattern Relationships:")
    for pattern_id, pattern in system.memory.patterns.items():
        print(f"\nPattern {pattern_id[:8]}:")
        print(f"├── Content: {pattern['content'][:50]}...")
        print(f"├── Strength: {pattern['strength']*100:.1f}%")
        print(f"├── Type: {pattern['analysis']['type']}")
        print(f"└── Relationships: {len(pattern['relationships'])}")
        
        if pattern['relationships']:
            print("    └── Connected to:")
            for rel_id in pattern['relationships']:
                rel_pattern = system.memory.patterns[rel_id]
                print(f"        ├── {rel_id[:8]}: {rel_pattern['content'][:30]}...")
    
    # Show cognitive metrics
    print("\n🧠 Final Cognitive State:")
    for metric, value in system.cognition.cognitive_metrics.items():
        print(f"├── {metric}: {value*100:.1f}%")
    
    # Verify learning occurred
    assert len(system.memory.patterns) > 0, "No patterns learned"
    assert system.memory.memory_metrics['integration_level'] > 0, "No integration occurred"
    assert system.cognition.cognitive_metrics['fpu_level'] > 0.0001, "No FPU growth"

if __name__ == "__main__":
    asyncio.run(test_cognitive_learning()) 