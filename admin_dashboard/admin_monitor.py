import sys
import os
import psutil
import time
import json
from flask import Flask, render_template, request, jsonify

# Initialize FractiCody with minimal knowledge
class FractiCognition:
    def __init__(self):
        self.memory = []  # Persistent knowledge base
        self.cognition_level = 1.0  # Starting intelligence level
        self.learning_active = True  # Allow learning by default

    def store_interaction(self, user_input, response):
        """Stores interactions dynamically for recursive learning."""
        self.memory.append({"input": user_input, "response": response, "timestamp": time.time()})

    def retrieve_last(self):
        """Retrieves the most recent stored interaction."""
        return self.memory[-1] if self.memory else None

    def process_input(self, user_input):
        """Recursive AI cognition - expands based on past interactions."""
        last_interaction = self.retrieve_last()
        
        if last_interaction:
            past_input, past_response = last_interaction["input"], last_interaction["response"]
            response = f"Building from '{past_input}', I have learned: {past_response}."
        else:
            response = "I am forming my initial understanding..."

        # Increase cognition level gradually
        self.cognition_level += 0.1
        response = f"[Cognition Level {self.cognition_level:.2f}] {response}"

        # Store new learning
        self.store_interaction(user_input, response)
        return response

# Initialize Components
app = Flask(__name__)
fracti_ai = FractiCognition()

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
