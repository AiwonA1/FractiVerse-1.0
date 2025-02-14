"""
⚡ FractiCody Processing Unit (FPU) Manager
Handles computational scaling & dynamic resource allocation.
"""
class FractiFPU:
    def __init__(self, base_power=1.0):
        self.processing_power = base_power

    def scale_fpu(self, factor):
        """Dynamically scales AI processing power based on demand."""
        self.processing_power *= factor
        return f"🚀 FPU Scaled to: {self.processing_power}x"

    def reset_fpu(self):
        """Resets FPU scaling to base power."""
        self.processing_power = 1.0
        return "🔄 FPU Reset to Base Power"

if __name__ == "__main__":
    fpu = FractiFPU()
    print(fpu.scale_fpu(2.5))
    print(fpu.reset_fpu())
