"""
📊 FractiAdmin 1.0 - Admin Monitoring Dashboard
Monitors system performance, AI processing units, and blockchain transactions.
"""

import sys
import os
import uvicorn
from fastapi import FastAPI

# Ensure the 'core' directory is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.fracti_fpu import FractiProcessingUnit  # ✅ Fixed Import Path

# Initialize FastAPI app
app = FastAPI()

class FractiAdminMonitor:
    def __init__(self):
        self.fpu = FractiProcessingUnit()
        self.system_metrics = {
            "CPU Usage": "N/A",
            "Memory Usage": "N/A",
            "Active AI Nodes": "N/A",
            "FractiChain Transactions": "N/A"
        }

    def update_system_metrics(self):
        """Gathers real-time system performance statistics from FractiFPU."""
        self.system_metrics["CPU Usage"] = f"{self.fpu.get_cpu_usage()}%"
        self.system_metrics["Memory Usage"] = f"{self.fpu.get_memory_usage()}GB"
        self.system_metrics["Active AI Nodes"] = self.fpu.get_active_nodes()
        self.system_metrics["FractiChain Transactions"] = self.fpu.get_transaction_count()

    def get_metrics(self):
        """Returns system metrics as a dictionary."""
        self.update_system_metrics()
        return self.system_metrics

# Create an instance of the admin monitor
admin_monitor = FractiAdminMonitor()

@app.get("/")
def home():
    """Admin UI API Home"""
    return {
        "message": "✅ FractiAdmin 1.0 Running",
        "system_metrics": admin_monitor.get_metrics(),
    }

@app.get("/metrics")
def get_metrics():
    """Fetch real-time system metrics."""
    return admin_monitor.get_metrics()

if __name__ == "__main__":
    # Set port dynamically from Render (default: 8181)
    port = int(os.environ.get("PORT", 8181))
    uvicorn.run(app, host="0.0.0.0", port=port)
