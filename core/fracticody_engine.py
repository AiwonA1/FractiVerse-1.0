import sys
import os
import psutil
import time
import json

# Ensure the script finds the 'core' directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import core components
from fractal_cognition import FractiCognition
from memory_manager import MemoryManager
from fracti_fpu import FractiProcessingUnit

class FractiCodyEngine:
    """Core engine for FractiCody AI"""
    
    def __init__(self):
        self.cognition = FractiCognition()
        self.memory = MemoryManager()
        self.fpu = FractiProcessingUnit()
        self.cognition_level = 1.0
        self.learning_active = True  # Enables deep learning
    
    def process_input(self, user_input):
        """Processes user input using fractal cognition"""
        memory_data = self.memory.retrieve_last()
        
        if memory_data:
            past_input, past_response = memory_data["input"], memory_data["response"]
            response = f"Building from '{past_input}', I have learned: {past_response}"
        else:
            response = "I am forming my initial understanding..."
        
        # Enhance cognition level dynamically
        self.cognition_level += 0.1
        response = f"[Cognition Level {self.cognition_level:.2f}] {response}"
        
        # Apply FractiProcessingUnit optimizations
        optimized_response = self.fpu.optimize_response(response)
        
        # Store learning data
        self.memory.store_interaction(user_input, optimized_response)
        
        return optimized_response
    
    def activate_deep_learning(self, status=True):
        """Enables or disables deep learning mode"""
        self.learning_active = status
        return "Deep Learning Activated." if status else "Deep Learning Paused."

# Initialize the engine instance globally
fracticody = FractiCodyEngine()

@app.route('/command', methods=['POST'])
def command():
    """Processes AI commands through FractiCody's cognition."""
    user_input = request.json.get("command", "").strip()
    response = fracticody.process_input(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=True)
