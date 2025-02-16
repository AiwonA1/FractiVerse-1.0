from flask import Flask, render_template, request, jsonify
import psutil
import time

# Simulated FractiChain & FPU (Replace with real implementations)
class FractiChain:
    memory_store = []
    
    def store(self, user_input, response):
        """Stores interaction in FractiChain memory."""
        self.memory_store.append({"user_input": user_input, "response": response, "timestamp": time.time()})
    
    def retrieve_last(self):
        """Retrieves the last stored interaction."""
        return self.memory_store[-1] if self.memory_store else None

class FractiProcessingUnit:
    def optimize_response(self, response):
        """Dummy optimizer to adjust responses dynamically."""
        return response + " [Optimized]"

# Initialize Components
app = Flask(__name__)
fractichain = FractiChain()
fpu = FractiProcessingUnit()

def fracticody_cognition(user_input):
    """Processes user input recursively for dynamic AI responses."""
    memory = fractichain.retrieve_last()
    
    responses = {
        "hello": "Hello! I am FractiCody. How can I assist you?",
        "status": "System is running optimally. All metrics are stable.",
        "fracti": "FractiVerse is active. What would you like to explore?",
        "memory": f"Memory Usage: {psutil.virtual_memory().percent}%",
        "cpu": f"CPU Usage: {psutil.cpu_percent()}%",
    }
    
    # Generate dynamic response
    response = responses.get(user_input.lower(), f"I am processing your request: {user_input}")
    
    # Optimize with FPU
    optimized_response = fpu.optimize_response(response)
    
    # Store in FractiChain
    fractichain.store(user_input, optimized_response)
    
    return optimized_response

@app.route('/')
def dashboard():
    metrics = {
        "CPU Usage": f"{psutil.cpu_percent()}%",
        "Memory Usage": f"{psutil.virtual_memory().total / (1024**3):.2f}GB",
        "Active AI Nodes": 9,  # Placeholder
        "FractiChain Transactions": len(fractichain.memory_store)  # Tracks stored conversations
    }
    return render_template("admin_dashboard.html", metrics=metrics)

@app.route('/command', methods=['POST'])
def command():
    user_input = request.json.get("command", "").strip()
    response = fracticody_cognition(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=True)
