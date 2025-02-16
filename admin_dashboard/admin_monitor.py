import sys
import os
import psutil
import time
from flask import Flask, render_template, request, jsonify

# Ensure Python finds `core` directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))

from fractal_cognition import FractiCognition  # Corrected Import

# Initialize Components
app = Flask(__name__)
fracti_ai = FractiCognition()  # Fractal Cognition Engine

def get_system_metrics():
    """Fetches real-time system statistics."""
    return {
        "CPU Usage": f"{psutil.cpu_percent()}%",
        "Memory Usage": f"{psutil.virtual_memory().total / (1024**3):.2f}GB",
        "Active AI Nodes": 9,  # Placeholder (expand later)
        "FractiChain Transactions": len(fracti_ai.memory)  # Tracks AI's memory depth
    }

@app.route('/')
def dashboard():
    """Loads the admin dashboard with live metrics."""
    metrics = get_system_metrics()
    return render_template("admin_dashboard.html", metrics=metrics)

@app.route('/command', methods=['POST'])
def command():
    """Processes AI commands through FractiCody's cognition."""
    user_input = request.json.get("command", "").strip().lower()
    
    # Special commands for recursive learning & deep cognition
    if user_input in ["begin deep learning", "start deep learning"]:
        fracti_ai.learning_active = True
        response = "Deep Learning Activated. I will continuously optimize and improve."
    elif user_input in ["stop deep learning", "pause deep learning"]:
        fracti_ai.learning_active = False
        response = "Deep Learning Paused."
    elif user_input in ["optimize cognition", "increase intelligence"]:
        fracti_ai.cognition_level += 0.5
        response = f"Cognition optimized. New cognition level: {fracti_ai.cognition_level:.2f}"
    elif user_input in ["what did i just say?", "recall last input"]:
        last_memory = fracti_ai.retrieve_last()
        response = f"Last recorded interaction: {last_memory}" if last_memory else "I have no prior memory stored yet."
    else:
        # Standard AI processing
        response = fracti_ai.process_input(user_input)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=True)
