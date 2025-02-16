from flask import Flask, render_template, request, jsonify
import psutil

# Initialize Flask App
app = Flask(__name__)

# AI Response Simulation (Replace with Real AI Integration)
def fracticody_ai(command):
    responses = {
        "hello": "Hello! I am FractiCody. How can I assist you?",
        "status": "System is running optimally. All metrics are stable.",
        "fracti": "FractiVerse is active. What would you like to explore?",
        "memory": f"Memory Usage: {psutil.virtual_memory().percent}%",
        "cpu": f"CPU Usage: {psutil.cpu_percent()}%",
    }
    
    return responses.get(command.lower(), f"Unknown command: {command}")

@app.route('/')
def dashboard():
    metrics = {
        "CPU Usage": f"{psutil.cpu_percent()}%",
        "Memory Usage": f"{psutil.virtual_memory().total / (1024**3):.2f}GB",
        "Active AI Nodes": 9,  # Placeholder for AI node tracking
        "FractiChain Transactions": 0  # Placeholder for FractiChain transactions
    }
    return render_template("admin_dashboard.html", metrics=metrics)

@app.route('/command', methods=['POST'])
def command():
    user_input = request.json.get("command", "").strip().lower()
    response = fracticody_ai(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=True)
