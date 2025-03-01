import torch
import numpy as np
from .pattern_processor import PatternProcessor
from .resonance_detector import ResonanceDetector
from .knowledge_integrator import KnowledgeIntegrator

class CognitiveController:
    """Control cognitive processing and growth"""
    def __init__(self):
        self.processor = PatternProcessor()
        self.resonance = ResonanceDetector()
        self.knowledge = KnowledgeIntegrator()
        self.fpu_level = 0.0001  # Start at 0.01%
        
    async def process_input(self, input_data):
        """Process input through cognitive system"""
        try:
            # Process through unipixel field
            patterns = self.processor.process_pattern(input_data)
            
            # Detect resonance
            resonance = self.resonance.detect_resonance(
                self.processor.emergence_history
            )
            
            # Integrate knowledge
            integrated = self.knowledge.integrate_patterns(resonance)
            
            # Generate response
            response = self._generate_response(integrated)
            
            return response
            
        except Exception as e:
            return f"Processing error: {str(e)}"
            
    async def expand_capacity(self, target_fpu):
        """Expand cognitive capacity"""
        try:
            while self.fpu_level < target_fpu:
                # Generate growth patterns
                patterns = self._generate_growth_patterns()
                
                # Process patterns
                processed = self.processor.process_pattern(patterns)
                
                # Check resonance
                resonance = self.resonance.detect_resonance(
                    self.processor.emergence_history
                )
                
                # Integrate knowledge
                if self.knowledge.integrate_patterns(resonance):
                    # Tiny growth increment
                    self.fpu_level += 0.0001  # 0.01% growth
                    
                yield {
                    "fpu_level": self.fpu_level,
                    "patterns": len(processed),
                    "resonance": len(resonance)
                }
                
        except Exception as e:
            yield f"Expansion error: {str(e)}"

    def _generate_growth_patterns(self):
        """Generate patterns for cognitive growth"""
        patterns = []
        complexity = self.fpu_level * 1000  # Scale complexity with FPU level
        
        # Generate basic patterns
        patterns.extend(self.processor.generate_patterns(complexity))
        
        # Add harmonic patterns
        harmonics = self._generate_harmonics(patterns[0])
        patterns.extend(harmonics)
        
        return patterns

    def _generate_harmonics(self, base_pattern):
        """Generate harmonic patterns"""
        harmonics = []
        
        # Generate 3 levels of harmonics
        for i in range(3):
            scale = (i + 2) / 2.0  # Harmonic scaling
            harmonic = torch.sin(scale * torch.asin(base_pattern))
            harmonics.append(harmonic)
        
        return harmonics

    def _generate_response(self, integrated_patterns):
        """Generate response from integrated patterns"""
        if not integrated_patterns:
            return "No coherent patterns detected"
        
        # Find strongest pattern
        strongest = max(integrated_patterns, 
                       key=lambda p: p.get('coherence', 0))
        
        # Format response
        response = {
            'pattern_strength': strongest.get('coherence', 0),
            'pattern_size': strongest.get('size', 0),
            'center': strongest.get('center', (0,0)),
            'stability': self.resonance.resonance_threshold
        }
        
        return response

    def get_status(self):
        """Get current cognitive status"""
        return {
            'fpu_level': self.fpu_level,
            'active_patterns': len(self.processor.emergence_history),
            'knowledge_size': len(self.knowledge.knowledge_graph),
            'resonance_threshold': self.resonance.resonance_threshold
        } 