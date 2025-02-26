from typing import List, Tuple, Optional
import numpy as np
from ..operators.unipixel_ops import UnipixelOperator

class UnipixelCore:
    """Core class for handling unipixel operations in the FractiVerse system."""
    
    def __init__(self, dimensions: Tuple[int, int, int] = (64, 64, 64)):
        """Initialize the UnipixelCore with given dimensions.
        
        Args:
            dimensions: Tuple of (width, height, depth) for the unipixel space
        """
        self.dimensions = dimensions
        self.space = np.zeros(dimensions, dtype=np.float32)
        self.operator = UnipixelOperator()
        self.active = False
        
    def start(self) -> bool:
        """Start the unipixel core system.
        
        Returns:
            bool: True if started successfully
        """
        try:
            self.active = True
            return True
        except Exception as e:
            print(f"Failed to start UnipixelCore: {e}")
            return False
            
    def stop(self) -> bool:
        """Stop the unipixel core system.
        
        Returns:
            bool: True if stopped successfully
        """
        try:
            self.active = False
            return True
        except Exception as e:
            print(f"Failed to stop UnipixelCore: {e}")
            return False
            
    def process_point(self, x: int, y: int, z: int, value: float) -> bool:
        """Process a single point in the unipixel space.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            value: Value to set at the point
            
        Returns:
            bool: True if point was processed successfully
        """
        if not self.active:
            return False
            
        try:
            if 0 <= x < self.dimensions[0] and \
               0 <= y < self.dimensions[1] and \
               0 <= z < self.dimensions[2]:
                self.space[x, y, z] = value
                return True
            return False
        except Exception as e:
            print(f"Failed to process point: {e}")
            return False
            
    def get_point(self, x: int, y: int, z: int) -> Optional[float]:
        """Get the value at a point in the unipixel space.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            
        Returns:
            Optional[float]: Value at the point or None if invalid coordinates
        """
        try:
            if 0 <= x < self.dimensions[0] and \
               0 <= y < self.dimensions[1] and \
               0 <= z < self.dimensions[2]:
                return float(self.space[x, y, z])
            return None
        except Exception as e:
            print(f"Failed to get point: {e}")
            return None
            
    def clear(self) -> bool:
        """Clear the unipixel space.
        
        Returns:
            bool: True if cleared successfully
        """
        try:
            self.space.fill(0)
            return True
        except Exception as e:
            print(f"Failed to clear space: {e}")
            return False

    def status(self) -> str:
        """Get the current status of the UnipixelCore.
        
        Returns:
            str: Current status ("active" or "inactive")
        """
        return "active" if self.active else "inactive"
        
    def get_metrics(self) -> dict:
        """Get current metrics for the UnipixelCore.
        
        Returns:
            dict: Dictionary containing current metrics
        """
        try:
            total_points = np.prod(self.dimensions)
            active_points = np.count_nonzero(self.space)
            density = float(active_points) / total_points
            
            return {
                "dimensions": self.dimensions,
                "total_points": int(total_points),
                "active_points": int(active_points),
                "density": density,
                "memory_usage": self.space.nbytes,
                "status": self.status()
            }
        except Exception as e:
            print(f"Failed to get metrics: {e}")
            return {
                "error": str(e),
                "status": self.status()
            } 