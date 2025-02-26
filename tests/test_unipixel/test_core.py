"""Tests for Unipixel core operations."""
import pytest
import numpy as np
from fractiverse.unipixel.core import (
    UnipixelCore,
    SpaceTime,
    Particle,
    Field
)

@pytest.fixture
async def unipixel_core():
    """Provide a test instance of UnipixelCore."""
    core = UnipixelCore(dimensions=3, test_mode=True)
    await core.initialize()
    try:
        yield core
    finally:
        await core.shutdown()

@pytest.fixture
def test_particle():
    """Create a test particle."""
    return Particle(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([1.0, 0.0, 0.0]),
        mass=1.0
    )

@pytest.fixture
def test_field():
    """Create a test field."""
    return Field(
        field_type="gravitational",
        strength=1.0,
        falloff=lambda r: 1/r**2 if r > 0 else 1.0
    )

@pytest.mark.asyncio
async def test_particle_creation(unipixel_core):
    """Test particle creation in space."""
    # Create particle
    particle = await unipixel_core.create_particle(
        position=np.array([1.0, 1.0, 1.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        mass=2.0
    )
    
    assert particle is not None
    assert np.allclose(particle.position, [1.0, 1.0, 1.0])
    assert particle.mass == 2.0
    
    # Verify particle is in space
    particles = await unipixel_core.get_particles()
    assert len(particles) == 1
    assert particles[0].id == particle.id

@pytest.mark.asyncio
async def test_field_interaction(unipixel_core, test_particle, test_field):
    """Test field-particle interactions."""
    # Add particle and field
    particle_id = await unipixel_core.add_particle(test_particle)
    field_id = await unipixel_core.add_field(test_field)
    
    # Run single timestep
    await unipixel_core.step_simulation(dt=0.1)
    
    # Get updated particle
    updated_particle = await unipixel_core.get_particle(particle_id)
    
    # Verify field effect
    assert not np.allclose(updated_particle.velocity, test_particle.velocity)
    assert np.all(np.isfinite(updated_particle.position))
    assert np.all(np.isfinite(updated_particle.velocity))

@pytest.mark.asyncio
async def test_space_expansion(unipixel_core):
    """Test dynamic space expansion."""
    initial_bounds = await unipixel_core.get_space_bounds()
    
    # Add particle near boundary
    await unipixel_core.create_particle(
        position=initial_bounds[1] * 0.9,  # Near positive boundary
        velocity=np.array([1.0, 1.0, 1.0]),
        mass=1.0
    )
    
    # Run simulation until space expands
    for _ in range(10):
        await unipixel_core.step_simulation(dt=0.1)
    
    new_bounds = await unipixel_core.get_space_bounds()
    assert np.any(new_bounds[1] > initial_bounds[1])

@pytest.mark.asyncio
async def test_energy_conservation(unipixel_core, test_particle):
    """Test energy conservation in closed systems."""
    # Add particle
    await unipixel_core.add_particle(test_particle)
    
    # Get initial energy
    initial_energy = await unipixel_core.get_total_energy()
    
    # Run simulation
    for _ in range(10):
        await unipixel_core.step_simulation(dt=0.01)
    
    # Get final energy
    final_energy = await unipixel_core.get_total_energy()
    
    # Check conservation (within numerical precision)
    assert np.isclose(final_energy, initial_energy, rtol=1e-10)

@pytest.mark.parametrize("field_type,expected_behavior", [
    ("gravitational", "attractive"),
    ("electromagnetic", "repulsive"),
    ("quantum", "probabilistic")
])
@pytest.mark.asyncio
async def test_field_behaviors(unipixel_core, field_type, expected_behavior):
    """Test different field behaviors."""
    # Create test field
    field = Field(
        field_type=field_type,
        strength=1.0,
        falloff=lambda r: 1/r**2 if r > 0 else 1.0
    )
    field_id = await unipixel_core.add_field(field)
    
    # Add test particles
    p1 = await unipixel_core.create_particle(
        position=np.array([-1.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        mass=1.0
    )
    
    p2 = await unipixel_core.create_particle(
        position=np.array([1.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        mass=1.0
    )
    
    # Run simulation
    await unipixel_core.step_simulation(dt=0.1)
    
    # Get updated particles
    p1_updated = await unipixel_core.get_particle(p1.id)
    p2_updated = await unipixel_core.get_particle(p2.id)
    
    if expected_behavior == "attractive":
        # Particles should move closer
        assert np.linalg.norm(p1_updated.position - p2_updated.position) < 2.0
    elif expected_behavior == "repulsive":
        # Particles should move apart
        assert np.linalg.norm(p1_updated.position - p2_updated.position) > 2.0
    else:  # probabilistic
        # Just verify movement occurred
        assert not np.allclose(p1_updated.position, p1.position) or \
               not np.allclose(p2_updated.position, p2.position) 