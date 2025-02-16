import sys
import os
import time
import json
from flask import Flask, request, jsonify

# Ensure Python detects the 'core' module properly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import core components
try:
    from core.fractal_cognition import FractalCognition
    from core.memory_manager import MemoryManager
    from core.fracti_fpu import FractiProcessingUnit
except ImportError as e:
    print(f"❌ Error importing core modules: {e}")
    raise

# Initialize Flask
app = Flask(__name__)

class FractiCodyEngine:
    """Core engine for FractiCody AI"""
    
    def __init__(self):
        print("🚀 Initializing FractiCody Engine...")
        self.fractal_cognition = FractalCognition()  # Bootstraps deep learning on init
        self.memory = MemoryManager()
        self.fpu = FractiProcessingUnit()
        self.cognition_level = self.fractal_cognition.cognition_level
        self.learning_active = True

    def start(self):
        """Starts the AI engine"""
        print("🔹 Activating Fractal Cognition...")
        self.fractal_cognition.activate()
        print(f"✅ FractiCody Booted at Cognition Level: {self.cognition_level}")

    def process_input(self, user_input):
        """Processes user input using fractal cognition"""
        return self.fractal_cognition.process_input(user_input)

@app.route('/command', methods=['POST'])
def command():
    """Processes AI commands through FractiCody's cognition."""
    try:
        user_input = request.json.get("command", "").strip()
        if not user_input:
            return jsonify({"error": "Invalid input. Command is required."}), 400

        fracticody = FractiCodyEngine()
        response = fracticody.process_input(user_input)
        return jsonify({"response": response})

    except Exception as e:
        print(f"❌ Error processing command: {e}")
        return jsonify({"error": "An error occurred processing your request."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=True)
