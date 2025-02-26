from typing import Dict, Optional
import numpy as np

class RealitySystem:
    """System for handling reality operations in the FractiVerse system."""
    
    def __init__(self):
        """Initialize the RealitySystem."""
        self.active = False
        self.state = {}
        self.reality_matrix = np.zeros((64, 64, 64), dtype=np.float32)
        
    def start(self) -> bool:
        """Start the reality system.
        
        Returns:
            bool: True if started successfully
        """
        try:
            self.active = True
            return True
        except Exception as e:
            print(f"Failed to start RealitySystem: {e}")
            return False
            
    def stop(self) -> bool:
        """Stop the reality system.
        
        Returns:
            bool: True if stopped successfully
        """
        try:
            self.active = False
            return True
        except Exception as e:
            print(f"Failed to stop RealitySystem: {e}")
            return False
            
    def process(self, input_data: Dict) -> Optional[Dict]:
        """Process input through the reality system.
        
        Args:
            input_data: Dictionary containing input data to process
            
        Returns:
            Optional[Dict]: Processed output data or None if processing failed
        """
        if not self.active:
            return None
            
        try:
            # Update reality matrix based on input
            if 'coordinates' in input_data:
                for coord in input_data['coordinates']:
                    x, y, z = coord['position']
                    value = coord.get('value', 1.0)
                    if 0 <= x < 64 and 0 <= y < 64 and 0 <= z < 64:
                        self.reality_matrix[x, y, z] = value
                        
            # Update state with summary statistics
            matrix_sum = float(np.sum(self.reality_matrix))
            matrix_mean = float(np.mean(self.reality_matrix))
            matrix_std = float(np.std(self.reality_matrix))
            
            # Get non-zero coordinates for efficient output
            non_zero_coords = np.nonzero(self.reality_matrix)
            sparse_matrix = [
                {
                    'position': [int(x), int(y), int(z)],
                    'value': float(self.reality_matrix[x, y, z])
                }
                for x, y, z in zip(*non_zero_coords)
            ]
            
            self.state.update({
                'last_input': input_data,
                'matrix_sum': matrix_sum,
                'matrix_mean': matrix_mean,
                'matrix_std': matrix_std,
                'active_points': len(sparse_matrix)
            })
            
            return {
                'reality_state': self.state,
                'sparse_matrix': sparse_matrix
            }
            
        except Exception as e:
            print(f"Failed to process input: {e}")
            return None
            
    def is_active(self) -> bool:
        """Check if the reality system is active.
        
        Returns:
            bool: True if the system is active
        """
        return self.active
        
    def reset(self) -> bool:
        """Reset the reality system to its initial state.
        
        Returns:
            bool: True if reset successfully
        """
        try:
            self.state = {}
            self.reality_matrix.fill(0)
            return True
        except Exception as e:
            print(f"Failed to reset RealitySystem: {e}")
            return False
