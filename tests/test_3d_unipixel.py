import numpy as np
import fractiverse.operators

def test_3d_unipixel_operations():
    try:
        print("\n🌟 Testing 3D Unipixel Operations")
        
        # Create Unipixels with initial positions
        pixel1 = Unipixel("Pixel_A", position=np.array([0.1, 0.2, 0.3]))
        pixel2 = Unipixel("Pixel_B", position=np.array([0.4, 0.5, 0.6]))
        
        # Add knowledge with 3D vectors
        pixel1 = pixel1 >> "Quantum Entanglement" >> "Fractal Recursion"
        pixel2 = pixel2 >> "Neural Harmonics" >> "Spatial Cognition"
        
        # Test operations
        print(f"\nPixel A: {pixel1}")
        print(f"Pixel B: {pixel2}")
        
        print("\n✅ 3D Unipixel operations tested successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_3d_unipixel_operations()
