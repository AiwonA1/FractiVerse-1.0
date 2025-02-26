"""Tests for visualization rendering engines."""
import pytest
import numpy as np
from pathlib import Path
from fractiverse.visualization.renderers import (
    FractalRenderer,
    UnipixelRenderer,
    TimelineRenderer,
    NetworkRenderer
)

@pytest.fixture
def test_data():
    """Provide test data for visualization."""
    return {
        "points": np.random.rand(100, 3),
        "colors": np.random.rand(100, 4),
        "connections": [(i, i+1) for i in range(99)]
    }

@pytest.fixture
def output_dir(tmp_path):
    """Provide temporary output directory."""
    return tmp_path / "viz_output"

def test_fractal_rendering(test_data, output_dir):
    """Test fractal visualization rendering."""
    renderer = FractalRenderer()
    
    # Generate fractal data
    fractal_data = np.random.rand(256, 256)
    colormap = "viridis"
    
    # Render fractal
    output_file = output_dir / "fractal.png"
    result = renderer.render(
        data=fractal_data,
        colormap=colormap,
        output_file=output_file
    )
    
    assert result.success
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_unipixel_rendering(test_data, output_dir):
    """Test unipixel space visualization."""
    renderer = UnipixelRenderer()
    
    # Render 3D space
    output_file = output_dir / "unipixel_space.png"
    result = renderer.render_space(
        points=test_data["points"],
        colors=test_data["colors"],
        output_file=output_file,
        camera_position=np.array([2.0, 2.0, 2.0])
    )
    
    assert result.success
    assert output_file.exists()
    assert output_file.stat().st_size > 0
    assert result.metadata["num_points"] == len(test_data["points"])

def test_timeline_rendering(test_data, output_dir):
    """Test timeline visualization."""
    renderer = TimelineRenderer()
    
    # Generate timeline data
    events = [
        {"time": t, "value": np.sin(t/10)} 
        for t in range(100)
    ]
    
    # Render timeline
    output_file = output_dir / "timeline.svg"
    result = renderer.render_timeline(
        events=events,
        output_file=output_file,
        title="Test Timeline"
    )
    
    assert result.success
    assert output_file.exists()
    assert output_file.suffix == ".svg"
    assert result.metadata["num_events"] == len(events)

def test_network_rendering(test_data, output_dir):
    """Test network graph visualization."""
    renderer = NetworkRenderer()
    
    # Render network
    output_file = output_dir / "network.pdf"
    result = renderer.render_network(
        nodes=test_data["points"],
        edges=test_data["connections"],
        output_file=output_file,
        layout="force_directed"
    )
    
    assert result.success
    assert output_file.exists()
    assert output_file.suffix == ".pdf"
    assert result.metadata["num_nodes"] == len(test_data["points"])
    assert result.metadata["num_edges"] == len(test_data["connections"])

@pytest.mark.parametrize("format,dpi", [
    ("png", 300),
    ("svg", None),
    ("pdf", 600)
])
def test_export_formats(test_data, output_dir, format, dpi):
    """Test different export formats and settings."""
    renderer = FractalRenderer()  # Use fractal renderer for format tests
    
    # Generate simple test data
    data = np.random.rand(100, 100)
    
    # Render in specified format
    output_file = output_dir / f"test_export.{format}"
    result = renderer.render(
        data=data,
        output_file=output_file,
        dpi=dpi
    )
    
    assert result.success
    assert output_file.exists()
    assert output_file.suffix == f".{format}"
    
    if dpi is not None:
        assert result.metadata["dpi"] == dpi 