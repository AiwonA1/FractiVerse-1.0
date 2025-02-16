from flask import Flask, render_template
import psutil

app = Flask(__name__)

def get_system_metrics():
    """Fetch live system metrics."""
    return {
        "CPU Usage": f"{psutil.cpu_percent()}%",
        "Memory Usage": f"{round(psutil.virtual_memory().total / (1024 ** 3), 2)}GB",
        "Active AI Nodes": 9,  # Placeholder - replace with live data
        "FractiChain Transactions": 0  # Placeholder - replace with live blockchain data
    }

@app.route("/")
def dashboard():
    metrics = get_system_metrics()
    return render_template("admin_dashboard.html", metrics=metrics)  # ✅ Pass metrics

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8181, debug=True)
