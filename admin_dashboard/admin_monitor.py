"""
📊 FractiAdmin 1.0 - Admin Monitoring Dashboard
Monitors system performance, AI processing units, and blockchain transactions.
"""

import sys
import os

# Ensure the 'core' directory is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.fracti_fpu import FractiProcessingUnit  # ✅ Fixed Import Path

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

    def display_metrics(self):
        """Displays live system metrics in the admin dashboard."""
        self.update_system_metrics()
        print("📊 FractiAdmin 1.0 - System Monitoring")
        for key, value in self.system_metrics.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    admin = FractiAdminMonitor()
    admin.display_metrics()
