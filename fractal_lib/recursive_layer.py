import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import numpy as np

class RecursiveLayer(nn.Module):
    """
    Implements a self-similar neural layer with recursive processing capabilities.
    Enables deep recursive learning through fractal pattern recognition.
    """
    
    def __init__(
        self,
        input_dim: int,
        recursion_levels: int = 3,
        fractal_dimension: float = 1.5,
        self_connection_strength: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.recursion_levels = recursion_levels
        self.fractal_dimension = fractal_dimension
        
        # Create recursive processing units
        self.recursive_units = nn.ModuleList([
            self._create_recursive_unit(input_dim)
            for _ in range(recursion_levels)
        ])
        
        # Self-connection matrix
        self.self_connections = nn.Parameter(
            torch.eye(input_dim) * self_connection_strength
        )
        
        # Fractal pattern gates
        self.pattern_gates = nn.Parameter(
            torch.randn(recursion_levels, input_dim)
        )
        
    def _create_recursive_unit(self, dim: int) -> nn.Module:
        """Creates a single recursive processing unit."""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_patterns: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass with recursive pattern processing.
        """
        batch_size = x.shape[0]
        current_state = x
        pattern_states = []
        
        for level in range(self.recursion_levels):
            # Apply self-connections
            self_connected = torch.matmul(
                current_state,
                self.self_connections
            )
            
            # Process through recursive unit
            processed = self.recursive_units[level](self_connected)
            
            # Apply fractal pattern gating
            pattern_gate = torch.sigmoid(self.pattern_gates[level])
            gated_state = processed * pattern_gate
            
            pattern_states.append(gated_state)
            current_state = gated_state
            
        # Combine patterns using fractal dimension weighting
        weights = torch.pow(
            torch.arange(self.recursion_levels, device=x.device),
            -self.fractal_dimension
        )
        weights = weights / weights.sum()
        
        output = sum(
            state * w for state, w in zip(pattern_states, weights)
        )
        
        if return_patterns:
            return output, pattern_states
        return output 