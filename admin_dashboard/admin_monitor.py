from flask import Flask, render_template, request, jsonify
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

@app.route("/command", methods=["POST"])
def process_command():
    """Process user commands for FractiCody and FractiEcosystem components."""
    command = request.json.get("command", "").strip().lower()

    # ✅ Add real command handling here
    if command == "status":
        response = "✅ FractiCody is operational."
    elif command == "reboot":
        response = "♻️ System reboot initiated..."
    elif command == "check chain":
        response = f"🔗 FractiChain Transactions: {get_system_metrics()['FractiChain Transactions']}"
    else:
        response = f"⚠️ Unknown command: {command}"

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8181, debug=True)
