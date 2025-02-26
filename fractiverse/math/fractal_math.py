"""Fractal mathematics operations module."""
import numpy as np

def mandelbrot_set(points, max_iter=100, escape_radius=2.0):
    """Calculate Mandelbrot set membership for given points.
    
    Args:
        points (np.ndarray): Array of complex points to test
        max_iter (int): Maximum number of iterations
        escape_radius (float): Radius beyond which points are considered escaped
        
    Returns:
        np.ndarray: Array of iteration counts for each point
    """
    points = points[:, 0] + points[:, 1]*1j
    z = np.zeros_like(points)
    n = np.zeros(points.shape, dtype=int)
    
    for i in range(max_iter):
        mask = np.abs(z) <= escape_radius
        z[mask] = z[mask]**2 + points[mask]
        n[mask] += 1
    
    return n

def julia_set(points, c, max_iter=100, escape_radius=2.0):
    """Calculate Julia set membership for given points.
    
    Args:
        points (np.ndarray): Array of complex points to test
        c (complex): Julia set parameter
        max_iter (int): Maximum number of iterations
        escape_radius (float): Radius beyond which points are considered escaped
        
    Returns:
        np.ndarray: Array of iteration counts for each point
    """
    z = points[:, 0] + points[:, 1]*1j
    n = np.zeros(points.shape[0], dtype=int)
    
    for i in range(max_iter):
        mask = np.abs(z) <= escape_radius
        z[mask] = z[mask]**2 + c
        n[mask] += 1
    
    return n

def fractal_dimension(points, eps=None):
    """Calculate the fractal dimension of a set of points using box-counting method.
    
    Args:
        points (np.ndarray): Array of points
        eps (float, optional): Box size. If None, multiple sizes are used.
        
    Returns:
        float: Estimated fractal dimension
    """
    if eps is None:
        eps = np.logspace(-2, 0, 20)
    
    counts = []
    for e in eps:
        boxes = np.floor(points/e)
        counts.append(len(np.unique(boxes, axis=0)))
    
    counts = np.array(counts)
    valid = counts > 0
    if np.sum(valid) < 2:
        return 1.0
    
    slope, _ = np.polyfit(np.log(eps[valid]), np.log(counts[valid]), 1)
    return -slope

def iterate_function(func, z0, c, iterations):
    """Iterate a function for fractal generation.
    
    Args:
        func (callable): Function to iterate
        z0 (complex): Initial value
        c (complex): Parameter value
        iterations (int): Number of iterations
        
    Returns:
        list: Sequence of iteration values
    """
    sequence = [z0]
    z = z0
    
    for _ in range(iterations):
        z = func(z, c)
        sequence.append(z)
    
    return sequence 