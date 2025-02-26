import numpy as np
from typing import List, Tuple, Optional

class Unipixel:
    def __init__(self, identifier, position=None, knowledge=[]):
        self.id = identifier
        self.knowledge = set(knowledge)
        self.position = position if position is not None else np.random.rand(3)
        self.vectors = {}

    def __rshift__(self, new_knowledge):
        self.knowledge.add(new_knowledge)
        return self

    def __repr__(self):
        return f"Unipixel({self.id}, Position: {self.position}, Knowledge: {list(self.knowledge)})"

class UnipixelOperator:
    """Operator class for handling unipixel operations."""
    
    def __init__(self):
        """Initialize the UnipixelOperator."""
        self.dimensions = (64, 64, 64)
        
    def validate_point(self, x: int, y: int, z: int) -> bool:
        """Validate if a point is within the unipixel space.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            
        Returns:
            bool: True if point is valid
        """
        return (0 <= x < self.dimensions[0] and
                0 <= y < self.dimensions[1] and
                0 <= z < self.dimensions[2])
                
    def calculate_density(self, points: List[Tuple[int, int, int]]) -> float:
        """Calculate the density of points in the unipixel space.
        
        Args:
            points: List of (x, y, z) coordinates
            
        Returns:
            float: Density value between 0 and 1
        """
        if not points:
            return 0.0
            
        total_volume = np.prod(self.dimensions)
        point_count = len(points)
        
        return point_count / total_volume
        
    def get_neighbors(self, x: int, y: int, z: int) -> List[Tuple[int, int, int]]:
        """Get valid neighboring points in the unipixel space.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            
        Returns:
            List[Tuple[int, int, int]]: List of valid neighboring coordinates
        """
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                        
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if self.validate_point(nx, ny, nz):
                        neighbors.append((nx, ny, nz))
                        
        return neighbors
        
    def interpolate_value(self, p1: Tuple[int, int, int], p2: Tuple[int, int, int],
                         v1: float, v2: float, t: float) -> float:
        """Interpolate a value between two points.
        
        Args:
            p1: First point coordinates
            p2: Second point coordinates
            v1: Value at first point
            v2: Value at second point
            t: Interpolation parameter (0 to 1)
            
        Returns:
            float: Interpolated value
        """
        return v1 + (v2 - v1) * t
