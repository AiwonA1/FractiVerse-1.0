"""
⚙️ FractiCody FPU - Adaptive Processing Unit
Manages real-time scaling, cognitive load balancing, system metrics, and efficiency optimizations.
"""

import random
import time
import psutil  # ✅ Added for system resource tracking

class FractiProcessingUnit:
    def __init__(self):
        self.base_fpu_capacity = 100  # Baseline processing power units
        self.current_load = 0  # Real-time cognitive load
        self.scale_factor = 1.0  # Adjusts dynamically
        self.optimization_threshold = 0.85  # Triggers efficiency optimizations
        self.active_nodes = 12  # Placeholder value for active AI nodes
        self.fracti_chain_transactions = 158  # Placeholder for FractiChain transactions

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

    def recursive_load_balancing(self, iterations=3):
        """Distributes workload recursively to maintain smooth AI performance."""
        if iterations == 0:
            return "⚖️ Load Balancing Complete"

        adjustment = random.uniform(0.9, 1.1)
        self.scale_factor *= adjustment
        time.sleep(0.5)  # Simulating processing
        return self.recursive_load_balancing(iterations - 1)

    def get_fpu_status(self):
        """Returns the current FPU performance statistics."""
        return {
            "Base FPU Capacity": self.base_fpu_capacity,
            "Current Load": self.current_load,
            "Scaling Factor": self.scale_factor,
        }

    ## ✅ New System Metrics Methods (Matching Admin Panel Expectations)

    def get_cpu_usage(self):
        """Returns the current system CPU usage percentage."""
        return psutil.cpu_percent(interval=1)

    def get_memory_usage(self):
        """Returns the current system memory usage in GB."""
        mem_info = psutil.virtual_memory()
        return round(mem_info.used / (1024 ** 3), 2)  # Convert bytes to GB

    def get_active_nodes(self):
        """Returns the number of active AI processing nodes."""
        return self.active_nodes

    def get_fractichain_transactions(self):
        """Returns the number of recorded FractiChain transactions."""
        return self.fracti_chain_transactions

if __name__ == "__main__":
    fpu = FractiProcessingUnit()
    print(fpu.adjust_fpu_load(80))  # Example task complexity
    print(fpu.optimize_performance())
    print(fpu.recursive_load_balancing())
    print(fpu.get_fpu_status())
    print(f"🖥️ CPU Usage: {fpu.get_cpu_usage()}%")
    print(f"💾 Memory Usage: {fpu.get_memory_usage()} GB")
    print(f"🤖 Active AI Nodes: {fpu.get_active_nodes()}")
    print(f"🔗 FractiChain Transactions: {fpu.get_fractichain_transactions()}")
