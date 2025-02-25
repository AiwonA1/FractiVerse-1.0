import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np

class FractalStoryteller(nn.Module):
    """
    Implements fractal-based narrative generation with recursive story patterns
    and self-similar plot structures.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        narrative_complexity: float = 0.7
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.narrative_complexity = narrative_complexity
        
        # Story pattern embeddings
        self.pattern_embedding = nn.Linear(embedding_dim, embedding_dim)
        
        # Recursive narrative layers
        self.narrative_layers = nn.ModuleList([
            self._create_narrative_layer()
            for _ in range(num_layers)
        ])
        
        # Plot structure attention
        self.plot_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            batch_first=True
        )
        
        # Narrative coherence gate
        self.coherence_gate = nn.Parameter(
            torch.ones(embedding_dim)
        )
        
    def _create_narrative_layer(self) -> nn.Module:
        """Creates a single narrative processing layer."""
        return nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.LayerNorm(self.embedding_dim * 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Dropout(0.1)
        )
        
    def forward(
        self,
        context: torch.Tensor,
        return_plot_structure: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Generates fractal narrative patterns from context.
        """
        batch_size = context.shape[0]
        
        # Embed initial story patterns
        patterns = self.pattern_embedding(context)
        plot_structures = []
        
        # Process through narrative layers
        for layer in self.narrative_layers:
            # Apply narrative processing
            narrative = layer(patterns)
            
            # Apply plot attention
            attended_plot, _ = self.plot_attention(
                narrative,
                narrative,
                narrative
            )
            
            # Generate self-similar plot structure
            plot_structure = self._generate_plot_structure(attended_plot)
            plot_structures.append(plot_structure)
            
            patterns = narrative + attended_plot
            
        # Apply narrative coherence
        coherence = torch.sigmoid(self.coherence_gate)
        final_narrative = patterns * coherence
        
        if return_plot_structure:
            return final_narrative, {
                'plot_structures': plot_structures,
                'coherence': coherence.detach().cpu().numpy()
            }
        return final_narrative
        
    def _generate_plot_structure(
        self,
        narrative: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates self-similar plot structure from narrative patterns.
        """
        # Extract key narrative elements
        narrative_elements = torch.chunk(
            narrative,
            int(1 / self.narrative_complexity),
            dim=-1
        )
        
        # Create fractal plot structure
        plot_structure = []
        for i, element in enumerate(narrative_elements):
            # Add self-similar variations
            variations = torch.stack([
                element,
                torch.roll(element, shifts=i, dims=-1),
                torch.flip(element, dims=[-1])
            ])
            plot_structure.append(variations.mean(0))
            
        return torch.cat(plot_structure, dim=-1) 