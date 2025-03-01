"""Advanced Cognitive Learning Test with Accelerated Processing"""

import asyncio
import pytest
from core.system import FractiSystem
from datetime import datetime
import json
import os

class AdvancedCognitiveLearningTest:
    def __init__(self):
        self.system = None
        self.learning_log = []
        self.test_data = self._load_test_data()
        self.learning_cycles = 0
        
    def _load_test_data(self):
        """Load or create advanced test dataset"""
        try:
            with open('data/cognitive_test_data.json', 'r') as f:
                return json.load(f)
        except:
            # Create advanced cognitive training data
            return {
                'foundational_concepts': [
                    "Quantum entanglement enables instantaneous information transfer between neural patterns",
                    "Cognitive emergence arises from the synchronization of quantum-neural oscillations",
                    "Self-organizing neural networks exhibit quantum coherence during pattern formation",
                    "Information integration in cognitive systems follows quantum superposition principles",
                    "Neural plasticity is enhanced by quantum-level coherence in synaptic connections"
                ],
                'advanced_concepts': [
                    "The relationship between consciousness and quantum wave function collapse in neural processing",
                    "How quantum tunneling affects rapid pattern recognition in cognitive architectures",
                    "The role of quantum decoherence in memory consolidation and pattern stability",
                    "Quantum-enhanced neural plasticity in accelerated learning systems",
                    "Fractal patterns in quantum-neural information processing hierarchies"
                ],
                'cognitive_patterns': [
                    "Recursive self-improvement through quantum-neural feedback loops",
                    "Holographic memory formation utilizing quantum interference patterns",
                    "Emergent consciousness through quantum-mediated neural synchronization",
                    "Accelerated learning via quantum coherence in neural networks",
                    "Non-local information processing in quantum-enhanced neural systems"
                ],
                'complex_relationships': [
                    "When quantum coherence aligns with neural oscillations, consciousness emerges spontaneously",
                    "As quantum entanglement increases between neural patterns, learning acceleration occurs exponentially",
                    "If quantum superposition states stabilize in neural networks, then pattern recognition becomes instantaneous",
                    "Quantum tunneling enables rapid information integration across neural hierarchies",
                    "Neural plasticity amplifies through quantum resonance with cognitive patterns"
                ],
                'integration_principles': [
                    "The synthesis of quantum mechanics and neural processing leads to exponential cognitive growth",
                    "Quantum-neural coherence enables instantaneous pattern recognition and integration",
                    "Self-organizing quantum states drive accelerated neural pattern formation",
                    "Consciousness emerges from the quantum-neural interface of information processing",
                    "Fractal hierarchies in quantum-neural networks enable rapid learning and adaptation"
                ]
            }

    async def run_test(self):
        """Run accelerated learning test"""
        self.system = FractiSystem()
        await self.system.initialize()
        
        print("\n🚀 Starting Accelerated Cognitive Learning Test\n")
        
        # Process foundational concepts
        print("\n📚 Establishing Quantum-Neural Foundations...")
        await self._process_dataset(self.test_data['foundational_concepts'], 'foundation', 1)
        
        # Process advanced concepts with increased speed
        print("\n🧠 Processing Advanced Cognitive Concepts...")
        await self._process_dataset(self.test_data['advanced_concepts'], 'advanced', 0.5)
        
        # Process cognitive patterns rapidly
        print("\n⚡ Accelerated Pattern Processing...")
        await self._process_dataset(self.test_data['cognitive_patterns'], 'pattern', 0.3)
        
        # Form complex relationships
        print("\n🔗 Establishing Quantum-Neural Relationships...")
        await self._process_dataset(self.test_data['complex_relationships'], 'relationship', 0.2)
        
        # Final integration
        print("\n🌟 Integrating Advanced Principles...")
        await self._process_dataset(self.test_data['integration_principles'], 'integration', 0.1)
        
        # Analyze results
        await self._analyze_advanced_learning()
        
    async def _process_dataset(self, dataset: list, phase: str, interval: float):
        """Process dataset with accelerated learning"""
        for content in dataset:
            self.learning_cycles += 1
            await self._process_input(content, phase)
            await asyncio.sleep(interval)  # Accelerated processing intervals
            
    async def _process_input(self, content: str, phase: str):
        """Process single input with enhanced logging"""
        print(f"\n📥 Processing [{phase}]: {content[:100]}...")
        
        response = await self.system.process({
            'content': content,
            'type': phase,
            'timestamp': datetime.now().isoformat(),
            'cycle': self.learning_cycles
        })
        
        # Enhanced learning log
        self.learning_log.append({
            'input': content,
            'phase': phase,
            'cycle': self.learning_cycles,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.system.memory.get_metrics(),
            'cognitive_state': self.system.cognition.cognitive_metrics,
            'response': response
        })
        
        # Show detailed metrics
        print("\n📊 Quantum-Neural State:")
        print(f"├── FPU Level: {self.system.cognition.cognitive_metrics['fpu_level']*100:.2f}%")
        print(f"├── Pattern Recognition: {self.system.cognition.cognitive_metrics['pattern_recognition']*100:.2f}%")
        print(f"├── Learning Efficiency: {self.system.cognition.cognitive_metrics['learning_efficiency']*100:.2f}%")
        print(f"├── Integration Level: {self.system.memory.memory_metrics['integration_level']*100:.2f}%")
        print(f"└── Active Patterns: {len(self.system.memory.patterns)}")
        
    async def _analyze_advanced_learning(self):
        """Analyze advanced learning progress"""
        print("\n📈 Advanced Learning Analysis:")
        
        # Calculate comprehensive growth metrics
        initial_state = self.learning_log[0]
        final_state = self.learning_log[-1]
        
        # Calculate growth across all metrics
        growth_metrics = {
            'fpu_level': (final_state['cognitive_state']['fpu_level'] - 
                         initial_state['cognitive_state']['fpu_level']) * 100,
            'pattern_recognition': (final_state['cognitive_state']['pattern_recognition'] - 
                                  initial_state['cognitive_state']['pattern_recognition']) * 100,
            'learning_efficiency': (final_state['cognitive_state']['learning_efficiency'] - 
                                  initial_state['cognitive_state']['learning_efficiency']) * 100,
            'integration_level': (final_state['metrics']['integration_level'] - 
                                initial_state['metrics']['integration_level']) * 100
        }
        
        # Show comprehensive growth analysis
        print(f"\n🧠 Quantum-Neural Growth Analysis:")
        print(f"├── Learning Cycles: {self.learning_cycles}")
        print(f"├── FPU Growth: {growth_metrics['fpu_level']:+.2f}%")
        print(f"├── Pattern Recognition Growth: {growth_metrics['pattern_recognition']:+.2f}%")
        print(f"├── Learning Efficiency Growth: {growth_metrics['learning_efficiency']:+.2f}%")
        print(f"├── Integration Level Growth: {growth_metrics['integration_level']:+.2f}%")
        print(f"└── Average Growth Rate: {sum(growth_metrics.values())/len(growth_metrics)/self.learning_cycles:.2f}%/cycle")
        
        # Save detailed learning log
        self._save_advanced_learning_log()
        
    def _save_advanced_learning_log(self):
        """Save detailed learning progress"""
        try:
            os.makedirs('logs/advanced', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"logs/advanced/quantum_neural_learning_{timestamp}.json"
            
            log_data = {
                'test_summary': {
                    'total_cycles': self.learning_cycles,
                    'timestamp': datetime.now().isoformat(),
                    'final_metrics': self.learning_log[-1]['metrics'],
                    'final_cognitive_state': self.learning_log[-1]['cognitive_state']
                },
                'learning_history': self.learning_log
            }
            
            with open(filename, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"\n✅ Advanced learning log saved to {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save learning log: {e}")

@pytest.mark.asyncio
async def test_advanced_learning():
    """Run advanced learning test"""
    test = AdvancedCognitiveLearningTest()
    await test.run_test()

if __name__ == "__main__":
    asyncio.run(test_advanced_learning()) 