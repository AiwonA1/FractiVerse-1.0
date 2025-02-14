"""
🌀 Fractal Cognition Engine - Core of FractiCody's Intelligence
Handles recursive AI computations, Unipixel intelligence, and fractal decision-making.
"""
from fracti_constants import MAX_INTELLIGENCE_DEPTH

class FractalCognition:
    def __init__(self):
        self.recursion_depth = 0
        self.intelligence_map = {}

    def activate(self):
        print("✅ Fractal Cognition Engine Activated")
        self.recursion_depth = 1

    def process_input(self, input_data):
        """Processes input recursively using fractal intelligence expansion."""
        if self.recursion_depth >= MAX_INTELLIGENCE_DEPTH:
            return input_data  # Stop recursion at max depth
        
        processed_data = f"🔄 Processed at Depth {self.recursion_depth}: {input_data}"
        self.recursion_depth += 1
        return self.process_input(processed_data)  # Recursive expansion

    def reset(self):
        """Resets fractal cognition depth."""
        self.recursion_depth = 0

if __name__ == "__main__":
    cognition = FractalCognition()
    cognition.activate()
    print(cognition.process_input("Fractal AI Test"))
