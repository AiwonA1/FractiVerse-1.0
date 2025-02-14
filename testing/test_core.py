"""
🧪 Test - Core Fractal AI Engine
Unit tests for Fractal Cognition, PEFF Harmonization, Memory, and FPU.
"""
import unittest
from core.fractal_cognition import FractalCognition
from core.peff_harmonization import PEFFHarmonization
from core.memory_manager import MemoryManager
from core.fracti_fpu import FractiFPU

class TestFractalAIEngine(unittest.TestCase):
    def test_fractal_cognition(self):
        cognition = FractalCognition()
        self.assertEqual(cognition.process_input("Test"), "🔄 Processed at Depth 1: Test")

    def test_peff_harmonization(self):
        peff = PEFFHarmonization()
        self.assertEqual(peff.adjust_harmony(1.5), "🔹 PEFF Harmony Level Adjusted to: 1.5")

    def test_memory_manager(self):
        memory = MemoryManager()
        memory.store_memory("AI Data")
        self.assertEqual(memory.retrieve_memory(), "📤 Retrieved Memory: AI Data")

    def test_fracti_fpu(self):
        fpu = FractiFPU()
        self.assertEqual(fpu.scale_fpu(2.0), "🚀 FPU Scaled to: 2.0x")

if __name__ == "__main__":
    unittest.main()
