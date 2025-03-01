"""
FractiVerse 1.0 Core System
Integrates all FractiVerse components into a unified system
"""

import asyncio
import torch
from typing import Dict, List, Optional
import time
import numpy as np

from .cognition.peff import PEFFSystem, SensoryInput
from .cognition.learning import LearningSystem
from .cognition.memory_integration import MemoryIntegration
from .cognition.memory_retrieval import MemoryRetrieval
from .cognition.pattern_completion import PatternCompletion
from .cognition.pattern_evolution import PatternEvolution
from .cognition.pattern_hybridization import PatternHybridization
from .cognition.pattern_emergence import PatternEmergence
from .cognition.pattern_analysis import PatternAnalysis

from .chain.blockchain import FractiChain
from .chain.consensus import FractalConsensus
from .network.protocol import FractiNet
from .treasury.management import FractiTreasury
from .token.contract import FractiToken

class FractiVerse:
    """Main FractiVerse system integration"""
    
    def __init__(self, dimensions: tuple = (256, 256)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dimensions = dimensions
        
        # Initialize components
        self.peff = PEFFSystem(dimensions)
        self.learning = LearningSystem(dimensions)
        
        # Memory systems
        self.memory = MemoryIntegration(dimensions)
        self.retrieval = MemoryRetrieval(self.memory)
        self.completion = PatternCompletion(self.retrieval, self.peff)
        
        # Pattern systems
        self.evolution = PatternEvolution(self.completion, self.peff)
        self.hybridization = PatternHybridization(self.evolution, self.peff)
        self.emergence = PatternEmergence(self.hybridization, self.peff)
        self.analysis = PatternAnalysis(self.emergence)
        
        # Blockchain components
        self.chain = FractiChain()
        self.consensus = FractalConsensus()
        
        # Network and economy
        self.network = FractiNet()
        self.treasury = FractiTreasury()
        self.token = FractiToken()
        
        # System metrics
        self.metrics = {
            'pattern_count': 0,
            'memory_usage': 0.0,
            'processing_time': 0.0,
            'system_coherence': 0.0
        }
        
        print("\n🌌 FractiVerse 1.0 Initialized")
        
    async def start(self):
        """Start FractiVerse system"""
        try:
            # Start network
            await self.network.start()
            
            # Start continuous processing
            asyncio.create_task(self._continuous_processing())
            
            print("\n✨ FractiVerse 1.0 Started")
            
        except Exception as e:
            print(f"Startup error: {e}")
            
    async def process_pattern(self, pattern: torch.Tensor, 
                            sensory_input: Optional[SensoryInput] = None) -> Dict:
        """Process new pattern through system"""
        try:
            start_time = time.time()
            
            # Process through PEFF
            if sensory_input:
                peff_coherence = self.peff.process_sensory_input(sensory_input)
            else:
                peff_coherence = 0.0
                
            # Learn pattern
            learned = await self.learning.learn_pattern(pattern, sensory_input)
            
            # Store in memory
            pattern_id = await self.memory.store_pattern(pattern, self.peff.metrics)
            
            # Analyze pattern
            analysis = await self.analysis.analyze_pattern(pattern)
            
            # Create blockchain record
            self.chain.add_pattern(pattern_id, pattern, analysis.emergence_indicators['emergence_score'])
            
            # Update metrics
            self._update_metrics(time.time() - start_time)
            
            return {
                'pattern_id': pattern_id,
                'peff_coherence': peff_coherence,
                'analysis': analysis,
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            print(f"Pattern processing error: {e}")
            return None
            
    async def _continuous_processing(self):
        """Continuous background processing"""
        while True:
            try:
                # Optimize memory
                await self.memory.optimize_memory()
                
                # Update emergence fields
                await self.emergence._update_emergence_fields(self.memory.holographic_field)
                
                # Process evolution
                if len(self.memory.patterns) > 0:
                    pattern = list(self.memory.patterns.values())[0].vector
                    evolved = await self.evolution.evolve_pattern(pattern)
                    if evolved:
                        await self.process_pattern(evolved.evolved_pattern)
                        
                # Distribute rewards
                self._distribute_rewards()
                
                await asyncio.sleep(1)  # Processing interval
                
            except Exception as e:
                print(f"Continuous processing error: {e}")
                await asyncio.sleep(5)  # Error recovery delay
                
    def _distribute_rewards(self):
        """Distribute rewards based on system metrics"""
        try:
            # Calculate rewards from treasury
            rewards = self.treasury.calculate_rewards(self.metrics['system_coherence'])
            
            # Distribute tokens
            for recipient, amount in rewards.items():
                self.token.transfer(
                    sender="treasury",
                    recipient=recipient,
                    amount=amount,
                    signature="system"
                )
                
        except Exception as e:
            print(f"Reward distribution error: {e}")
            
    def _update_metrics(self, processing_time: float):
        """Update system metrics"""
        self.metrics['pattern_count'] = len(self.memory.patterns)
        self.metrics['memory_usage'] = len(self.memory.patterns) / 1000  # Normalized to 0-1
        self.metrics['processing_time'] = processing_time
        self.metrics['system_coherence'] = np.mean([
            self.memory.metrics['avg_coherence'],
            self.peff.metrics['peff_alignment'],
            self.learning.metrics['pattern_coherence']
        ]) 