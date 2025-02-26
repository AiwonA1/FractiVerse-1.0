"""Tests for fractal mathematics operations."""
import pytest
import numpy as np
from fractiverse.math.fractal_math import (
    mandelbrot_set,
    julia_set,
    fractal_dimension,
    iterate_function
)

@pytest.fixture
def test_parameters():
    """Test parameters for fractal calculations."""
    return {
        "max_iter": 100,
        "escape_radius": 2.0,
        "precision": 1e-6
    }

def test_mandelbrot_set(test_parameters):
    """Test Mandelbrot set calculation."""
    # Test known points in the Mandelbrot set
    points = np.array([
        [0.0, 0.0],  # Center point (known to be in set)
        [0.25, 0.0],  # Real point in set
        [2.0, 2.0],   # Point definitely outside set
    ])
    
    results = mandelbrot_set(points, test_parameters["max_iter"], test_parameters["escape_radius"])
    
    assert results[0] == test_parameters["max_iter"]  # Center point should be in set
    assert results[1] == test_parameters["max_iter"]  # 0.25 should be in set
    assert results[2] < test_parameters["max_iter"]   # Outside point should escape

def test_julia_set(test_parameters):
    """Test Julia set calculation."""
    c = -0.4 + 0.6j  # Common Julia set parameter
    points = np.array([
        [0.0, 0.0],  # Origin
        [1.0, 0.0],  # Point likely outside set
        [0.5, 0.5]   # Test point
    ])
    
    results = julia_set(points, c, test_parameters["max_iter"], test_parameters["escape_radius"])
    
    assert isinstance(results, np.ndarray)
    assert results.shape == (3,)
    assert all(0 <= x <= test_parameters["max_iter"] for x in results)

def test_fractal_dimension():
    """Test fractal dimension calculation."""
    # Create a Koch curve-like pattern
    points = np.array([
        [0, 0],
        [1/3, 0],
        [0.5, np.sqrt(1/12)],  # Height of equilateral triangle
        [2/3, 0],
        [1, 0]
    ])
    
    dimension = fractal_dimension(points)
    
    # Koch curve has dimension ~1.26
    assert 1.2 < dimension < 1.3
    assert isinstance(dimension, float)
    
    # Test with a line (dimension should be close to 1)
    line_points = np.array([[x, 0] for x in np.linspace(0, 1, 10)])
    line_dimension = fractal_dimension(line_points)
    assert 0.9 < line_dimension < 1.1
    
    # Test with a filled square (dimension should be close to 2)
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    square_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    square_dimension = fractal_dimension(square_points)
    assert 1.9 < square_dimension < 2.1

def test_function_iteration():
    """Test function iteration for fractal generation."""
    def test_func(z, c):
        return z**2 + c
    
    z0 = 0.0
    c = 0.25
    iterations = 10
    
    result = iterate_function(test_func, z0, c, iterations)
    
    assert len(result) == iterations + 1  # Include initial point
    assert result[0] == z0
    assert isinstance(result, list)
    assert all(isinstance(x, (complex, float)) for x in result)

@pytest.mark.parametrize("input_value,expected", [
    (0.0, 0.0),
    (1.0, 1.0),
    (float('inf'), float('inf'))
])
def test_edge_cases(input_value, expected, test_parameters):
    """Test edge cases for fractal calculations."""
    point = np.array([[input_value, 0.0]])
    
    # Test Mandelbrot calculation with edge cases
    result = mandelbrot_set(point, test_parameters["max_iter"], test_parameters["escape_radius"])
    
    if np.isinf(expected):
        assert result[0] < test_parameters["max_iter"]
    else:
        assert isinstance(result[0], (np.int64, np.int32)) 