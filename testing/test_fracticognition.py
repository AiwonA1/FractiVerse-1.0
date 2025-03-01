import torch
import numpy as np
from core.fracticognition import FractiCognition

def test_pattern_emergence():
    """Test real pattern emergence and processing"""
    
    # Initialize system
    fracti = FractiCognition()
    
    # Test 1: Basic Pattern Recognition
    print("\nTest 1: Basic Pattern Recognition")
    input_text = "Hello FractiCognition"
    response = fracti.process(input_text)
    print(f"Input: {input_text}")
    print(f"Response: {response}")
    
    # Test 2: Pattern Evolution
    print("\nTest 2: Pattern Evolution")
    # Create a simple wave pattern
    x = np.linspace(0, 4*np.pi, 100)
    wave = np.sin(x) + np.sin(2*x)
    response = fracti.process(wave)
    print(f"Wave pattern response: {response}")
    
    # Test 3: Complex Pattern Interaction
    print("\nTest 3: Complex Pattern Interaction")
    # Create interfering patterns
    pattern1 = np.sin(x) * np.cos(x.reshape(-1, 1))
    pattern2 = np.cos(2*x) * np.sin(2*x.reshape(-1, 1))
    combined = pattern1 + pattern2
    response = fracti.process(combined)
    print(f"Complex pattern response: {response}")
    
    # Test 4: Memory and Learning
    print("\nTest 4: Memory and Learning")
    # Process same pattern multiple times
    for i in range(3):
        response = fracti.process(wave)
        print(f"Learning iteration {i+1}: {response}")
    
    # Test 5: Pattern Generation
    print("\nTest 5: Pattern Generation")
    # Generate response to learned pattern
    response = fracti.process(wave[:50])  # Partial pattern
    print(f"Generated completion: {response}")

if __name__ == "__main__":
    test_pattern_emergence() 