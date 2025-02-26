from typing import Dict, List, Optional
import numpy as np
from .reality_system import RealitySystem
from .peff_system import PeffSystem

class CognitiveEngine:
    """Core cognitive processing engine for the FractiVerse system."""
    
    def __init__(self):
        """Initialize the CognitiveEngine with its subsystems."""
        self.reality = RealitySystem()
        self.peff = PeffSystem()
        self.active = False
        self.cognitive_state = {}
        
    def start(self) -> bool:
        """Start the cognitive engine and its subsystems.
        
        Returns:
            bool: True if started successfully
        """
        try:
            if self.reality.start() and self.peff.start():
                self.active = True
                return True
            return False
        except Exception as e:
            print(f"Failed to start CognitiveEngine: {e}")
            return False
            
    def stop(self) -> bool:
        """Stop the cognitive engine and its subsystems.
        
        Returns:
            bool: True if stopped successfully
        """
        try:
            self.active = False
            peff_stopped = self.peff.stop()
            reality_stopped = self.reality.stop()
            return peff_stopped and reality_stopped
        except Exception as e:
            print(f"Failed to stop CognitiveEngine: {e}")
            return False
            
    def process_input(self, input_data: Dict) -> Optional[Dict]:
        """Process cognitive input through the engine.
        
        Args:
            input_data: Dictionary containing input data to process
            
        Returns:
            Optional[Dict]: Processed output data or None if processing failed
        """
        if not self.active:
            return None
            
        try:
            # Process through reality system
            reality_output = self.reality.process(input_data)
            if not reality_output:
                return None
                
            # Process through PEFF system
            peff_output = self.peff.process(reality_output)
            if not peff_output:
                return None
                
            # Update cognitive state
            self.cognitive_state.update({
                'last_input': input_data,
                'reality_state': reality_output,
                'peff_state': peff_output
            })
            
            return {
                'cognitive_state': self.cognitive_state,
                'output': peff_output
            }
        except Exception as e:
            print(f"Failed to process input: {e}")
            return None
            
    def get_state(self) -> Dict:
        """Get the current state of the cognitive engine.
        
        Returns:
            Dict: Current cognitive state
        """
        return {
            'active': self.active,
            'cognitive_state': self.cognitive_state,
            'reality_active': self.reality.is_active(),
            'peff_active': self.peff.is_active()
        }
        
    def reset(self) -> bool:
        """Reset the cognitive engine to its initial state.
        
        Returns:
            bool: True if reset successfully
        """
        try:
            self.cognitive_state = {}
            self.reality.reset()
            self.peff.reset()
            return True
        except Exception as e:
            print(f"Failed to reset CognitiveEngine: {e}")
            return False 