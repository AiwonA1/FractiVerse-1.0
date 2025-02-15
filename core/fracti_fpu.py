"""
⚙️ FractiProcessingUnit - FractiCody's Adaptive Processing Unit
Manages dynamic processing load, AI efficiency, and FractiChain transactions.
"""

import random
import time
import psutil  # ✅ Added for system metrics


class FractiProcessingUnit:
    def __init__(self):
        self.base_fpu_capacity = 100  # Baseline processing power units
        self.current_load = 0  # Real-time cognitive load
        self.scale_factor = 1.0  # Adjusts dynamically
        self.optimization_threshold = 0.85  # Triggers efficiency optimizations
        self.transaction_count = 0  # Tracks FractiChain transactions
        self.active_nodes = random.randint(5, 50)  # Simulated AI nodes count
        self.last_balancing_status = "⚖️ Load Balancing Pending"

    # -------------------------
    # ✅ FPU SCALING FUNCTIONS
    # -------------------------

    def adjust_fpu_load(self, task_complexity):
        """Dynamically scales FractiCody's processing power based on task demand."""
        self.current_load = min(1.0, task_complexity / self.base_fpu_capacity)
        self.scale_factor = max(0.5, min(2.0, 1.0 + (self.current_load - 0.5) * 1.5))
        return f"🔧 FPU Scaling Factor Adjusted: {self.scale_factor:.2f}"

    def optimize_performance(self):
        """Reduces unnecessary processing cycles if load is below the optimization threshold."""
        if self.current_load < self.optimization_threshold:
            self.scale_factor *= 0.9  # Reduce processing allocation
            return "🛠️ FPU Optimization Applied"
        return "✅ FPU Running at Peak Efficiency"

    # -------------------------
    # ✅ LOAD BALANCING SYSTEM
    # -------------------------

    def recursive_load_balancing(self, iterations=3):
        """Distributes workload recursively to maintain smooth AI performance."""
        if iterations == 0:
            self.last_balancing_status = "⚖️ Load Balancing Complete"
            return self.last_balancing_status

        adjustment = random.uniform(0.9, 1.1)
        self.scale_factor *= adjustment
        time.sleep(0.5)  # Simulating processing
        return self.recursive_load_balancing(iterations - 1)

    # -------------------------
    # ✅ SYSTEM METRICS & MONITORING
    # -------------------------

    def get_cpu_usage(self):
        """Returns the current CPU usage percentage."""
        return psutil.cpu_percent(interval=1)

    def get_memory_usage(self):
        """Returns the current memory usage in GB."""
        mem_info = psutil.virtual_memory()
        return round(mem_info.used / (1024 ** 3), 2)  # Convert bytes to GB

    def get_active_nodes(self):
        """Returns the number of active AI nodes in FractiCody's ecosystem."""
        return self.active_nodes

    # -------------------------
    # ✅ FRACTICHAIN TRANSACTIONS
    # -------------------------

    def get_transaction_count(self):
        """Returns the total number of transactions processed by FractiChain."""
        return self.transaction_count

    def increment_transaction_count(self):
        """Increments the FractiChain transaction count."""
        self.transaction_count += 1
        return f"🔄 Transaction Added | Total: {self.transaction_count}"

    # -------------------------
    # ✅ OVERALL SYSTEM STATUS
    # -------------------------

    def get_fpu_status(self):
        """Returns the current FPU performance statistics."""
        return {
            "Base FPU Capacity": self.base_fpu_capacity,
            "Current Load": self.current_load,
            "Scaling Factor": self.scale_factor,
            "CPU Usage": f"{self.get_cpu_usage()}%",
            "Memory Usage": f"{self.get_memory_usage()} GB",
            "Active AI Nodes": self.get_active_nodes(),
            "FractiChain Transactions": self.get_transaction_count(),
            "Last Load Balancing": self.last_balancing_status,
        }


if __name__ == "__main__":
    fpu = FractiProcessingUnit()
    print(fpu.adjust_fpu_load(80))  # Example task complexity
    print(fpu.optimize_performance())
    print(fpu.recursive_load_balancing())
    print(fpu.get_fpu_status())
    print(fpu.increment_transaction_count())  # Simulating a transaction update
