"""
⚙️ FractiCody FPU - Adaptive Processing Unit
Manages real-time scaling, cognitive load balancing, and efficiency optimizations.
"""

import random
import time
import psutil  # ✅ Required for system metrics

class FractiProcessingUnit:
    def __init__(self):
        self.base_fpu_capacity = 100  # Baseline processing power units
        self.current_load = 0  # Real-time cognitive load
        self.scale_factor = 1.0  # Adjusts dynamically
        self.optimization_threshold = 0.85  # Triggers efficiency optimizations

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

    ## ✅ NEW: System Monitoring Functions ##
    def get_cpu_usage(self):
        """Returns current CPU usage percentage."""
        return psutil.cpu_percent(interval=1)

    def get_memory_usage(self):
        """Returns system memory usage in GB."""
        mem = psutil.virtual_memory()
        return round(mem.used / (1024 ** 3), 2)

if __name__ == "__main__":
    fpu = FractiProcessingUnit()
    print(fpu.adjust_fpu_load(80))  # Example task complexity
    print(fpu.optimize_performance())
    print(fpu.recursive_load_balancing())
    print(fpu.get_fpu_status())
    print(f"CPU Usage: {fpu.get_cpu_usage()}%")
    print(f"Memory Usage: {fpu.get_memory_usage()} GB")
