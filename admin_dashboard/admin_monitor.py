"""
🛠 FractiAdmin - AI System Monitoring & Administration
Monitors and manages FractiCody ecosystem-wide performance.
"""

import os
import time
from core.fracti_fpu import FractiProcessingUnit

class AdminMonitor:
    def __init__(self):
        self.fpu = FractiProcessingUnit()
        self.system_metrics = {
            "CPU Usage": "0%",
            "Memory Usage": "0GB",
            "Active AI Nodes": 0,
            "FractiChain Transactions": 0
        }

    def update_system_metrics(self):
        """Updates live AI system metrics."""
        self.system_metrics["CPU Usage"] = f"{self.fpu.get_cpu_usage()}%"
        self.system_metrics["Memory Usage"] = f"{self.fpu.get_memory_usage()}GB"
        self.system_metrics["Active AI Nodes"] = self.fpu.get_active_nodes()
        self.system_metrics["FractiChain Transactions"] = self.fpu.get_transaction_count()

    def display_metrics(self):
        """Displays the latest system status."""
        self.update_system_metrics()
        print("📊 **FractiCody System Status:**")
        for key, value in self.system_metrics.items():
            print(f"🔹 {key}: {value}")

if __name__ == "__main__":
    admin = AdminMonitor()
    while True:
        admin.display_metrics()
        time.sleep(5)  # Refresh every 5 seconds
