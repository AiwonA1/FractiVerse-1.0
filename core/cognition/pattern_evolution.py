"""
Pattern Evolution System
Implements pattern evolution, mutation, and creative generation
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from .peff import PEFFSystem, SensoryInput
from .pattern_completion import CompletionResult

@dataclass
class EvolutionResult:
    """Result of pattern evolution"""
    evolved_pattern: torch.Tensor
    parent_patterns: List[str]
    mutation_score: float
    novelty_score: float
    peff_resonance: Dict[str, float]
    evolution_time: float

class PatternEvolution:
    """Advanced pattern evolution and creative generation"""
    
    def __init__(self, pattern_completion, peff_system: PEFFSystem):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.completion = pattern_completion
        self.peff = peff_system
        
        # Evolution parameters
        self.mutation_rate = 0.1
        self.crossover_points = 3
        self.population_size = 10
        self.generations = 5
        
        # Creative fields
        self.novelty_field = torch.zeros((256, 256), dtype=torch.complex64).to(self.device)
        self.evolution_memory: List[torch.Tensor] = []
        self.max_memory = 1000
        
        print("\n🧬 Pattern Evolution Initialized")
        
    async def evolve_pattern(self, base_pattern: torch.Tensor, 
                           target_peff_state: Optional[Dict[str, float]] = None) -> EvolutionResult:
        """Evolve pattern towards target PEFF state"""
        try:
            start_time = time.time()
            
            # Initialize population
            population = await self._initialize_population(base_pattern)
            
            # Evolution loop
            for generation in range(self.generations):
                # Evaluate fitness
                fitness_scores = self._evaluate_fitness(population, target_peff_state)
                
                # Selection
                parents = self._select_parents(population, fitness_scores)
                
                # Crossover and mutation
                population = await self._evolve_generation(parents)
                
            # Select best evolved pattern
            best_pattern = self._select_best(population, target_peff_state)
            
            # Calculate evolution metrics
            mutation_score = self._calculate_mutation_score(base_pattern, best_pattern)
            novelty = self._calculate_novelty(best_pattern)
            peff_resonance = self._calculate_peff_resonance(best_pattern)
            
            # Update evolution memory
            self._update_evolution_memory(best_pattern)
            
            return EvolutionResult(
                evolved_pattern=best_pattern,
                parent_patterns=[],  # TODO: Track parent pattern IDs
                mutation_score=mutation_score,
                novelty_score=novelty,
                peff_resonance=peff_resonance,
                evolution_time=time.time() - start_time
            )
            
        except Exception as e:
            print(f"Pattern evolution error: {e}")
            return None
            
    async def generate_creative_pattern(self, seed_pattern: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate new creative pattern"""
        try:
            # Start with seed or random pattern
            if seed_pattern is None:
                pattern = torch.randn(256, 256).to(self.device)
                pattern = pattern / torch.norm(pattern)
            else:
                pattern = seed_pattern.clone()
                
            # Apply creative transformations
            pattern = await self._apply_creative_transforms(pattern)
            
            # Enhance with PEFF
            pattern = self._enhance_with_peff(pattern)
            
            # Update novelty field
            self._update_novelty_field(pattern)
            
            return pattern
            
        except Exception as e:
            print(f"Creative generation error: {e}")
            return None
            
    async def _initialize_population(self, base_pattern: torch.Tensor) -> List[torch.Tensor]:
        """Initialize evolution population"""
        population = [base_pattern.clone()]
        
        # Generate variations
        for _ in range(self.population_size - 1):
            # Create mutation
            mutation = base_pattern.clone()
            mutation = await self._mutate_pattern(mutation)
            population.append(mutation)
            
        return population
        
    def _evaluate_fitness(self, population: List[torch.Tensor], 
                         target_state: Optional[Dict[str, float]]) -> List[float]:
        """Evaluate population fitness"""
        fitness_scores = []
        
        for pattern in population:
            # Calculate PEFF alignment
            if target_state:
                current_state = self.peff.process_sensory_input(pattern)
                alignment = self._calculate_state_alignment(current_state, target_state)
            else:
                alignment = 0.5  # Neutral alignment
                
            # Calculate novelty
            novelty = self._calculate_novelty(pattern)
            
            # Calculate coherence
            coherence = torch.mean(torch.abs(pattern)).item()
            
            # Combined fitness score
            fitness = (alignment + novelty + coherence) / 3
            fitness_scores.append(fitness)
            
        return fitness_scores
        
    def _select_parents(self, population: List[torch.Tensor], 
                       fitness_scores: List[float]) -> List[torch.Tensor]:
        """Select parents for next generation"""
        # Convert to probabilities
        probs = np.array(fitness_scores)
        probs = probs / probs.sum()
        
        # Select parents
        parent_indices = np.random.choice(
            len(population),
            size=len(population) // 2,
            p=probs,
            replace=False
        )
        
        return [population[i] for i in parent_indices]
        
    async def _evolve_generation(self, parents: List[torch.Tensor]) -> List[torch.Tensor]:
        """Create new generation through crossover and mutation"""
        new_population = parents.copy()
        
        while len(new_population) < self.population_size:
            # Select parent pair
            p1, p2 = np.random.choice(parents, size=2, replace=False)
            
            # Crossover
            child = await self._crossover_patterns(p1, p2)
            
            # Mutation
            child = await self._mutate_pattern(child)
            
            new_population.append(child)
            
        return new_population
        
    async def _crossover_patterns(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Perform pattern crossover"""
        child = torch.zeros_like(p1)
        
        # Generate crossover points
        points = sorted(np.random.choice(
            p1.shape[0],
            size=self.crossover_points,
            replace=False
        ))
        
        # Alternate segments
        start = 0
        use_p1 = True
        
        for point in points + [p1.shape[0]]:
            if use_p1:
                child[start:point] = p1[start:point]
            else:
                child[start:point] = p2[start:point]
            start = point
            use_p1 = not use_p1
            
        return child / torch.norm(child)
        
    async def _mutate_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        """Apply pattern mutation"""
        if torch.rand(1).item() > self.mutation_rate:
            return pattern
            
        # Generate mutation mask
        mask = torch.rand_like(pattern) < self.mutation_rate
        
        # Apply mutations
        mutations = torch.randn_like(pattern)
        pattern[mask] += mutations[mask]
        
        return pattern / torch.norm(pattern)
        
    async def _apply_creative_transforms(self, pattern: torch.Tensor) -> torch.Tensor:
        """Apply creative transformations"""
        # Apply fractal transformation
        pattern = self._apply_fractal_transform(pattern)
        
        # Apply quantum interference
        pattern = self._apply_quantum_interference(pattern)
        
        # Apply novelty field
        pattern = pattern + 0.1 * self.novelty_field
        
        return pattern / torch.norm(pattern)
        
    def _apply_fractal_transform(self, pattern: torch.Tensor) -> torch.Tensor:
        """Apply fractal transformation"""
        # Create fractal mask
        x = torch.linspace(-2, 2, pattern.shape[0]).to(self.device)
        y = torch.linspace(-2, 2, pattern.shape[1]).to(self.device)
        xx, yy = torch.meshgrid(x, y)
        r = torch.sqrt(xx**2 + yy**2)
        
        # Apply transformation
        transformed = pattern * torch.exp(-r)
        return transformed / torch.norm(transformed)
        
    def _apply_quantum_interference(self, pattern: torch.Tensor) -> torch.Tensor:
        """Apply quantum interference pattern"""
        # Generate interference
        x = torch.linspace(0, 2*np.pi, pattern.shape[0]).to(self.device)
        y = torch.linspace(0, 2*np.pi, pattern.shape[1]).to(self.device)
        xx, yy = torch.meshgrid(x, y)
        
        interference = torch.sin(xx) * torch.cos(yy)
        
        # Apply interference
        pattern = pattern + 0.1 * interference
        return pattern / torch.norm(pattern)
        
    def _update_novelty_field(self, pattern: torch.Tensor):
        """Update novelty field"""
        # Decay existing field
        self.novelty_field *= 0.99
        
        # Add new pattern influence
        self.novelty_field = (self.novelty_field + pattern) / 2
        self.novelty_field = self.novelty_field / torch.norm(self.novelty_field) 