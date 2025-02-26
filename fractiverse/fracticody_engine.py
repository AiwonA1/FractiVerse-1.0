from .operators import FractiVector, Unipixel, FractiChain, FractiNet

class FractiCognitiveEngine:
    def __init__(self):
        self.network = FractiNet()
        self.memory_chain = FractiChain()
        self.active_pixels = {}
        
    def process_thought(self, thought_text):
        """Process a thought using 3D cognitive vectors"""
        vector = FractiVector(thought_text)
        pixel = Unipixel(f"Thought_{len(self.active_pixels)}")
        pixel = pixel >> thought_text
        
        self.active_pixels[pixel.id] = pixel
        self.network = self.network | pixel
        self.memory_chain = self.memory_chain >> thought_text
        
        return vector
        
    def get_cognitive_state(self):
        """Return current cognitive state"""
        return {
            'active_pixels': len(self.active_pixels),
            'memory_chain_length': len(self.memory_chain.chain),
            'network_size': len(self.network.network)
        }
