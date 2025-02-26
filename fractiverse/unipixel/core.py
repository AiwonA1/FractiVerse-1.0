"""Unipixel core operations module."""
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional

@dataclass
class Particle:
    """Particle in unipixel space."""
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    id: Optional[str] = None

@dataclass
class Field:
    """Field in unipixel space."""
    field_type: str
    strength: float
    falloff: Callable[[float], float]
    id: Optional[str] = None

class SpaceTime:
    """Spacetime container for unipixel simulation."""
    
    def __init__(self, dimensions=3):
        """Initialize spacetime.
        
        Args:
            dimensions (int): Number of spatial dimensions
        """
        self.dimensions = dimensions
        self.bounds = np.array([
            [-10.0] * dimensions,
            [10.0] * dimensions
        ])
        
    def expand(self, factor=1.5):
        """Expand space bounds.
        
        Args:
            factor (float): Expansion factor
        """
        center = (self.bounds[1] + self.bounds[0]) / 2
        extent = (self.bounds[1] - self.bounds[0]) / 2
        self.bounds = np.array([
            center - extent * factor,
            center + extent * factor
        ])
        
    def contains(self, position):
        """Check if position is within bounds.
        
        Args:
            position (np.ndarray): Position to check
            
        Returns:
            bool: Whether position is within bounds
        """
        return np.all(position >= self.bounds[0]) and np.all(position <= self.bounds[1])

class UnipixelCore:
    """Core unipixel simulation engine."""
    
    def __init__(self, dimensions=3, test_mode=False):
        """Initialize unipixel core.
        
        Args:
            dimensions (int): Number of spatial dimensions
            test_mode (bool): Whether to run in test mode
        """
        self.dimensions = dimensions
        self.test_mode = test_mode
        self.space = SpaceTime(dimensions)
        self.particles: List[Particle] = []
        self.fields: List[Field] = []
        self.total_energy = 0.0
        self._next_id = 0
        
    async def initialize(self):
        """Initialize the simulation."""
        self.total_energy = self._compute_total_energy()
        
    async def shutdown(self):
        """Shutdown the simulation."""
        self.particles = []
        self.fields = []
        
    def _get_next_id(self):
        """Get next unique ID."""
        self._next_id += 1
        return str(self._next_id)
        
    async def create_particle(self, position, velocity, mass):
        """Create a new particle.
        
        Args:
            position (np.ndarray): Initial position
            velocity (np.ndarray): Initial velocity
            mass (float): Particle mass
            
        Returns:
            Particle: Created particle
        """
        particle = Particle(
            position=np.array(position),
            velocity=np.array(velocity),
            mass=mass,
            id=self._get_next_id()
        )
        self.particles.append(particle)
        return particle
        
    async def add_particle(self, particle):
        """Add existing particle to simulation.
        
        Args:
            particle (Particle): Particle to add
            
        Returns:
            str: Particle ID
        """
        if particle.id is None:
            particle.id = self._get_next_id()
        self.particles.append(particle)
        return particle.id
        
    async def add_field(self, field):
        """Add field to simulation.
        
        Args:
            field (Field): Field to add
            
        Returns:
            str: Field ID
        """
        if field.id is None:
            field.id = self._get_next_id()
        self.fields.append(field)
        return field.id
        
    async def get_particles(self):
        """Get all particles.
        
        Returns:
            list: List of particles
        """
        return self.particles
        
    async def get_particle(self, particle_id):
        """Get particle by ID.
        
        Args:
            particle_id (str): Particle ID
            
        Returns:
            Particle: Found particle or None
        """
        for p in self.particles:
            if p.id == particle_id:
                return p
        return None
        
    async def get_space_bounds(self):
        """Get current space bounds.
        
        Returns:
            np.ndarray: Space bounds
        """
        return self.space.bounds
        
    async def get_total_energy(self):
        """Get total system energy.
        
        Returns:
            float: Total energy
        """
        return self._compute_total_energy()
        
    def _compute_total_energy(self):
        """Compute total system energy."""
        energy = 0.0
        for p in self.particles:
            # Kinetic energy
            energy += 0.5 * p.mass * np.sum(p.velocity**2)
            
            # Potential energy from fields
            for f in self.fields:
                r = np.linalg.norm(p.position)
                if r > 0:
                    energy += p.mass * f.strength * f.falloff(r)
        
        return energy
        
    async def step_simulation(self, dt):
        """Step simulation forward.
        
        Args:
            dt (float): Time step
        """
        # Update particles
        for p in self.particles:
            # Apply field forces
            force = np.zeros(self.dimensions)
            for f in self.fields:
                r = np.linalg.norm(p.position)
                if r > 0:
                    direction = -p.position / r  # Normalize
                    magnitude = f.strength * f.falloff(r)
                    force += direction * magnitude * p.mass
            
            # Update velocity and position
            p.velocity += force * dt / p.mass
            p.position += p.velocity * dt
            
            # Check bounds
            if not self.space.contains(p.position):
                self.space.expand()
                
        # Update energy
        self.total_energy = self._compute_total_energy() 