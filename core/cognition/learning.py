"""
FractiVerse Learning System
Implements continuous learning with PEFF integration
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

from .peff import PEFFSystem, SensoryInput
from .processors.fractal import FractalProcessor
from .processors.quantum import QuantumCatalyst

@dataclass
class LearningPattern:
    """Pattern with learning metadata"""
    id: str
    data: torch.Tensor
    peff_state: Dict[str, float]
    coherence: float
    connections: List[str]
    timestamp: float

class LearningSystem:
    """Continuous learning system with PEFF integration"""
    
    def __init__(self, dimensions: tuple = (256, 256)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dimensions = dimensions
        
        # Initialize components
        self.peff = PEFFSystem(dimensions)
        self.fractal = FractalProcessor(dimensions)
        self.quantum = QuantumCatalyst(dimensions)
        
        # Learning state
        self.patterns: Dict[str, LearningPattern] = {}
        self.connection_strength = torch.zeros((1000, 1000)).to(self.device)  # Max 1000 patterns
        
        # Learning metrics
        self.metrics = {
            'learning_rate': 0.0001,
            'pattern_coherence': 0.0,
            'connection_density': 0.0,
            'peff_alignment': 0.0
        }
        
        print("\n🧠 Learning System Initialized")
        
    async def learn_pattern(self, pattern_data: torch.Tensor, sensory_input: Optional[SensoryInput] = None):
        """Learn new pattern with PEFF integration"""
        try:
            # Process through PEFF if sensory input available
            if sensory_input:
                peff_coherence = self.peff.process_sensory_input(sensory_input)
            else:
                peff_coherence = 0.0
                
            # Extract fractal features
            fractal_features = self.fractal.extract_fractal_features(pattern_data)
            
            # Apply quantum catalyst
            enhanced = self.quantum.apply_quantum_enhancement(pattern_data)
            
            # Create learning pattern
            pattern = LearningPattern(
                id=f"pattern_{len(self.patterns)}",
                data=enhanced,
                peff_state=self.peff.metrics.copy(),
                coherence=fractal_features['self_similarity'] * (1 + peff_coherence),
                connections=[],
                timestamp=time.time()
            )
            
            # Store pattern
            self.patterns[pattern.id] = pattern
            
            # Update connections
            await self._update_connections(pattern)
            
            # Update metrics
            self._update_metrics(pattern)
            
            return pattern
            
        except Exception as e:
            print(f"Learning error: {e}")
            return None
            
    async def _update_connections(self, new_pattern: LearningPattern):
        """Update pattern connections based on coherence"""
        if not self.patterns:
            return
            
        # Calculate similarities with existing patterns
        for pattern_id, pattern in self.patterns.items():
            if pattern_id == new_pattern.id:
                continue
                
            # Calculate similarity
            similarity = torch.cosine_similarity(
                new_pattern.data.flatten().unsqueeze(0),
                pattern.data.flatten().unsqueeze(0)
            ).item()
            
            # Update connection if similar enough
            if similarity > 0.7:  # Similarity threshold
                new_pattern.connections.append(pattern_id)
                pattern.connections.append(new_pattern.id)
                
                # Update connection strength matrix
                idx1 = int(new_pattern.id.split('_')[1])
                idx2 = int(pattern_id.split('_')[1])
                self.connection_strength[idx1, idx2] = similarity
                self.connection_strength[idx2, idx1] = similarity
                
    def _update_metrics(self, pattern: LearningPattern):
        """Update learning metrics"""
        # Update pattern coherence
        self.metrics['pattern_coherence'] = np.mean([p.coherence for p in self.patterns.values()])
        
        # Update connection density
        total_possible = len(self.patterns) * (len(self.patterns) - 1) / 2
        total_connections = sum(len(p.connections) for p in self.patterns.values()) / 2
        self.metrics['connection_density'] = total_connections / total_possible if total_possible > 0 else 0
        
        # Update PEFF alignment
        self.metrics['peff_alignment'] = pattern.peff_state['peff_alignment']
        
        # Adjust learning rate based on metrics
        self.metrics['learning_rate'] = min(
            0.01,  # Maximum learning rate
            self.metrics['learning_rate'] + 
            self.metrics['pattern_coherence'] * 0.0001 +
            self.metrics['peff_alignment'] * 0.0001
        ) 