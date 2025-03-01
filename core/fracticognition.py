from .cognitive_processor import CognitiveProcessor
from .unipixel_processor import UnipixelProcessor
from .pattern_emergence import PatternEmergence

class FractiCognition:
    def __init__(self):
        self.cognitive = CognitiveProcessor()
        self.memory = {}
        self.active_patterns = set()
        
    def process(self, input_data):
        """Process input through cognitive system"""
        # Process through cognitive system
        response = self.cognitive.process(input_data)
        
        # Update memory
        self._update_memory(response)
        
        # Generate output
        output = self._generate_output(response)
        
        return output 