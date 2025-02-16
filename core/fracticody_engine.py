import sys
import os
import psutil
import time
import json
from flask import Flask, request, jsonify
from core.fractal_cognition import FractiCognition  # Ensure this path is correct

# Initialize FractiCody Engine
class FractiCodyEngine:
    def __init__(self):
        self.fracti_ai = FractiCognition()  # Initialize Cognition Engine

    def process_input(self, user_input):
        """Handles user input and recursively learns."""
        response = self.fracti_ai.process_input(user_input)
        return response

# Flask API to interact with FractiCody
app = Flask(__name__)
fracticody = FractiCodyEngine()

@app.route('/command', methods=['POST'])
def command():
    """Processes AI commands through FractiCody's cognition."""
    user_input = request.json.get("command", "").strip()
    response = fracticody.process_input(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=True)
