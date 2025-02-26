from fractiverse.ui import UnipixelNavigator3D
import time

def demo_navigator():
    # Initialize 3D Navigator
    navigator = UnipixelNavigator3D()
    
    # Add some Unipixels with knowledge
    navigator.add_unipixel("Quantum_Node", [
        "Quantum Entanglement",
        "Wave Function Collapse"
    ])
    
    navigator.add_unipixel("AI_Node", [
        "Neural Networks",
        "Deep Learning"
    ])
    
    navigator.add_unipixel("PEFF_Node", [
        "Paradise Energy",
        "Fractal Force"
    ])
    
    # Run the navigator
    navigator.run()
    
    # Simulate cognitive updates
    for _ in range(10):
        time.sleep(1)
        navigator.update_cognitive_state()
        
    input("Press Enter to close...")

if __name__ == "__main__":
    demo_navigator()
