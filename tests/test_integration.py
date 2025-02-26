import unittest
from fractiverse.operators import FractiVector, Unipixel, FractiChain, FractiNet
from fractiverse.core import CognitiveEngine, RealitySystem, PeffSystem

class TestOperatorIntegration(unittest.TestCase):
    def setUp(self):
        self.cognitive_engine = CognitiveEngine()
        self.reality = RealitySystem()
        self.peff = PeffSystem()
        
    def test_vector_integration(self):
        """Test 3D Cognitive Vector integration"""
        vector = FractiVector("Test Thought")
        result = self.cognitive_engine.process_thought(vector)
        self.assertIsNotNone(result)
        
    def test_unipixel_integration(self):
        """Test Unipixel recursive processing"""
        pixel = Unipixel("Test_Pixel")
        pixel = pixel >> "Test Knowledge"
        self.assertIn("Test Knowledge", pixel.knowledge)
        
    def test_chain_integration(self):
        """Test FractiChain persistence"""
        chain = FractiChain()
        chain = chain >> "Test Memory"
        self.assertIn("Test Memory", chain.chain)
        
    def test_network_integration(self):
        """Test FractiNet distribution"""
        net = FractiNet()
        pixel = Unipixel("Test_Node")
        net = net | pixel
        self.assertIn("Test_Node", net.network)

if __name__ == '__main__':
    unittest.main()
