import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

class DeepRecursiveMemory(nn.Module):
    """
    Implements fractal-based deep memory structures with recursive storage
    and retrieval patterns.
    """
    
    def __init__(
        self,
        memory_dim: int,
        num_levels: int = 4,
        compression_ratio: float = 0.5,
        persistence_factor: float = 0.8
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_levels = num_levels
        self.persistence_factor = persistence_factor
        
        # Create recursive memory levels
        self.memory_levels = nn.ModuleList([
            self._create_memory_level(
                int(memory_dim * (compression_ratio ** i))
            )
            for i in range(num_levels)
        ])
        
        # Memory persistence gates
        self.persistence_gates = nn.Parameter(
            torch.ones(num_levels, memory_dim)
        )
        
        # Initialize memory states
        self.memory_states = [
            torch.zeros(1, dim)
            for dim in [int(memory_dim * (compression_ratio ** i))
                       for i in range(num_levels)]
        ]
        
    def _create_memory_level(self, dim: int) -> nn.Module:
        """Creates a single recursive memory level."""
        return nn.ModuleDict({
            'encoder': nn.Sequential(
                nn.Linear(self.memory_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU()
            ),
            'processor': nn.LSTM(
                dim,
                dim,
                num_layers=2,
                batch_first=True
            ),
            'decoder': nn.Sequential(
                nn.Linear(dim, self.memory_dim),
                nn.LayerNorm(self.memory_dim),
                nn.GELU()
            )
        })
        
    def forward(
        self,
        x: torch.Tensor,
        return_memory_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass implementing recursive memory processing.
        """
        batch_size = x.shape[0]
        current_input = x
        memory_outputs = []
        
        # Process through memory levels
        for level_idx, level in enumerate(self.memory_levels):
            # Encode input for this level
            encoded = level['encoder'](current_input)
            
            # Update memory state
            processed, new_state = level['processor'](
                encoded,
                None  # Use zero initial state
            )
            
            # Apply persistence
            persistence = torch.sigmoid(self.persistence_gates[level_idx])
            self.memory_states[level_idx] = (
                self.memory_states[level_idx] * persistence +
                processed * (1 - persistence)
            )
            
            # Decode memory state
            decoded = level['decoder'](self.memory_states[level_idx])
            memory_outputs.append(decoded)
            
            # Prepare input for next level
            current_input = decoded
            
        # Combine memory outputs with learned persistence
        persistence_weights = torch.softmax(
            torch.stack([g.mean() for g in self.persistence_gates]),
            dim=0
        )
        
        output = sum(
            out * w for out, w in zip(memory_outputs, persistence_weights)
        )
        
        if return_memory_states:
            return output, {
                'memory_states': self.memory_states,
                'persistence_weights': persistence_weights.detach().cpu().numpy()
            }
        return output
    
    def reset_memory(self):
        """Resets all memory states to zero."""
        self.memory_states = [
            torch.zeros_like(state) for state in self.memory_states
        ] 