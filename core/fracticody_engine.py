import sys
import os
import time
import json
from flask import Flask, request, jsonify

# Ensure Python detects the 'core' module properly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import core components (ensuring proper module detection)
try:
    from core.fractal_cognition import FractalCognition
    from core.memory_manager import MemoryManager
    from core.fracti_fpu import FractiProcessingUnit
except ImportError as e:
    print(f"Error importing core modules: {e}")
    raise

# Initialize Flask
app = Flask(__name__)

class FractiCodyEngine:
    """Core engine for FractiCody AI"""
    
    def __init__(self):
        print("Initializing FractiCody Engine...")
        self.cognition = FractalCognition()
        self.memory = MemoryManager()
        self.fpu = FractiProcessingUnit()
        self.cognition_level = 1.0
        self.learning_active = True  # Enables deep learning
        time.sleep(1)  # Prevents race conditions during initialization

    def process_input(self, user_input):
        """Processes user input using fractal cognition"""
        user_input = user_input.strip().lower()
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

# Initialize the engine inside Flask route to prevent premature execution
@app.route('/command', methods=['POST'])
def command():
    """Processes AI commands through FractiCody's cognition."""
    try:
        user_input = request.json.get("command", "").strip()
        if not user_input:
            return jsonify({"error": "Invalid input. Command is required."}), 400

        fracticody = FractiCodyEngine()  # ✅ Initialize only when called
        response = fracticody.process_input(user_input)
        return jsonify({"response": response})

    except Exception as e:
        print(f"Error processing command: {e}")
        return jsonify({"error": "An error occurred processing your request."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=True)
