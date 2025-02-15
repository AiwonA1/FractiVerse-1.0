"""
📊 FractiAdmin 1.0 - Admin Dashboard Monitor
Tracks system metrics, AI node activity, and FractiChain transactions.
"""

import os
import time  # ✅ Import added to fix NameError
import psutil
from core.fracti_fpu import FractiProcessingUnit

class FractiAdminMonitor:
    def __init__(self):
        self.fpu = FractiProcessingUnit()
        self.system_metrics = {
            "CPU Usage": "N/A",
            "Memory Usage": "N/A",
            "Active AI Nodes": "N/A",
            "FractiChain Transactions": "N/A",
        }

    def update_system_metrics(self):
        """Fetches real-time system metrics."""
        self.system_metrics["CPU Usage"] = f"{self.fpu.get_cpu_usage()}%"
        self.system_metrics["Memory Usage"] = f"{self.fpu.get_memory_usage()}GB"
        self.system_metrics["Active AI Nodes"] = self.fpu.get_active_nodes()
        self.system_metrics["FractiChain Transactions"] = self.fpu.get_fracti_transactions()

    def display_metrics(self):
        """Displays live system metrics for admins."""
        self.update_system_metrics()
        print("\n📊 **FractiCody System Status:**")
        for key, value in self.system_metrics.items():
            print(f"🔹 {key}: {value}")

if __name__ == "__main__":
    admin = FractiAdminMonitor()
    while True:
        admin.display_metrics()
        time.sleep(5)  # ✅ Fix: Now time.sleep() is recognized
