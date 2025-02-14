"""
📊 FractiCody Admin Monitor - AI Performance & Governance Tracking
Monitors system-wide AI activities, resource usage, and intelligence scaling.
"""
class AdminMonitor:
    def __init__(self):
        self.performance_metrics = {
            "CPU Usage": "15%",
            "Memory Usage": "2.5GB",
            "Active AI Nodes": 12,
            "FractiChain Transactions": 158
        }

    def get_system_status(self):
        """Returns an overview of FractiCody's current performance metrics."""
        status = "\n".join([f"{k}: {v}" for k, v in self.performance_metrics.items()])
        return f"📊 System Status:\n{status}"

    def update_metric(self, metric, value):
        """Updates a specific system performance metric."""
        if metric in self.performance_metrics:
            self.performance_metrics[metric] = value
            return f"✅ Updated {metric} to {value}"
        return "❌ Invalid Metric"

if __name__ == "__main__":
    monitor = AdminMonitor()
    print(monitor.get_system_status())
    print(monitor.update_metric("CPU Usage", "20%"))
