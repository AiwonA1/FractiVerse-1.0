"""
🧠 FractiAdmin - Cognitive Flow Visualization
Tracks real-time AI thought processes & Unipixel intelligence patterns.
"""

import time
import random
from core.fractal_cognition import FractalCognition
from blockchain.fracti_blockchain import FractiChain

class CognitionFlow:
    def __init__(self):
        self.cognition_engine = FractalCognition()
        self.blockchain = FractiChain()
        self.thought_log = []

    def capture_thought_process(self):
        """Captures real-time AI thought processes."""
        thought = self.cognition_engine.generate_cognitive_state()
        self.thought_log.append(thought)
        self.blockchain.store_thought(thought)
        return thought

    def display_cognitive_flow(self):
        """Displays the live AI cognition stream."""
        while True:
            thought_state = self.capture_thought_process()
            print(f"🧠 **FractiCody Thought Process:** {thought_state}")
            time.sleep(random.uniform(1.5, 3.0))  # Simulated thought intervals

if __name__ == "__main__":
    cognition_monitor = CognitionFlow()
    cognition_monitor.display_cognitive_flow()
