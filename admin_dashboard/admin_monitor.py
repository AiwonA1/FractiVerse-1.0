from flask import Flask, jsonify, render_template
import psutil
import requests
import os

app = Flask(__name__)

def get_system_metrics():
    """Fetch live system metrics"""
    return {
        "CPU Usage": f"{psutil.cpu_percent()}%",
        "Memory Usage": f"{round(psutil.virtual_memory().total / (1024 ** 3), 2)}GB",
        "Active AI Nodes": get_active_nodes(),
        "FractiChain Transactions": get_fractichain_transactions()
    }

def get_active_nodes():
    """Simulated function - Replace with actual AI node tracking"""
    return len(psutil.pids())  # Example: Counting active process IDs

def get_fractichain_transactions():
    """Fetch real FractiChain transaction count from an API"""
    FRACTICHAIN_API = "https://your-fractichain-api.com/transactions"
    try:
        response = requests.get(FRACTICHAIN_API, timeout=5)
        if response.status_code == 200:
            return response.json().get("transaction_count", 0)
        else:
            return "⚠️ API Error"
    except requests.exceptions.RequestException:
        return "⚠️ API Unreachable"

@app.route("/")
def dashboard():
    """Render the Admin Dashboard UI"""
    return render_template("admin_dashboard.html")

@app.route("/metrics")
def metrics():
    """API endpoint to get live system metrics"""
    return jsonify(get_system_metrics())

if __name__ == "__main__":
    port = int(os.environ.get("ADMIN_PORT", 8181))
    app.run(host="0.0.0.0", port=port, debug=True)
