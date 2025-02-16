from flask import Flask, render_template, jsonify
import psutil
import random

app = Flask(__name__)

def get_system_metrics():
    return {
        "CPU Usage": f"{psutil.cpu_percent()}%",
        "Memory Usage": f"{psutil.virtual_memory().used / (1024 ** 3):.2f}GB",
        "Active AI Nodes": random.randint(5, 50),
        "FractiChain Transactions": random.randint(0, 1000)
    }

@app.route("/")
def dashboard():
    metrics = get_system_metrics()
    return render_template("admin_dashboard.html", metrics=metrics)

@app.route("/metrics")
def metrics():
    return jsonify(get_system_metrics())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
