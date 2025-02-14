"""
🌀 FractiCody Engine - Core AI System
Handles Recursive Intelligence, Fractal Cognition, and Decision-Making
"""
from fracti_constants import FRACTICODY_VERSION
from FractiCody1_0.core.fractal_cognition import FractalCognition

class FractiCodyEngine:
    def __init__(self):
        self.fractal_cognition = FractalCognition()

    def start(self):
        print(f"🚀 FractiCody {FRACTICODY_VERSION} is initializing...")
        self.fractal_cognition.activate()

if __name__ == "__main__":
    engine = FractiCodyEngine()
    engine.start()
