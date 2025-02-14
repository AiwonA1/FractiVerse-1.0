"""
🚀 FractiCody 1.0 - Main Entry Point
Handles Fractal AI Initialization and System Coordination
"""
from fracticody_engine import FractiCodyEngine
from config import settings

if __name__ == "__main__":
    print("🔹 Initializing FractiCody 1.0...")
    fracti_ai = FractiCodyEngine()
    fracti_ai.start()
    print("✅ FractiCody 1.0 is now running.")
