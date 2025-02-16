import sys
import os
import psutil
import time
from flask import Flask, render_template, request, jsonify

# Ensure Python recognizes the 'core' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.fractal_cognition import FractiCognition  # Import FractiCody AI Engine

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
    user_input = request.json.get("command", "").strip()
    response = fracti_ai.process_input(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=True)
