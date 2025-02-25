import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np

class MultiScaleOperator(nn.Module):
    """
    Implements multi-scale fractal processing for handling patterns at different
    scales of complexity and abstraction.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_scales: int = 4,
        scale_factor: float = 2.0,
        attention_heads: int = 4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        
        # Create scale-specific processors
        self.scale_processors = nn.ModuleList([
            self._create_scale_processor(
                int(input_dim * (scale_factor ** i))
            )
            for i in range(num_scales)
        ])
        
        # Cross-scale attention
        self.cross_scale_attention = nn.MultiheadAttention(
            input_dim,
            attention_heads,
            batch_first=True
        )
        
        # Scale integration weights
        self.scale_weights = nn.Parameter(
            torch.ones(num_scales) / num_scales
        )
        
    def _create_scale_processor(self, dim: int) -> nn.Module:
        """Creates processing module for a specific scale."""
        return nn.Sequential(
            nn.Linear(self.input_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, self.input_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_scale_outputs: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass implementing multi-scale processing.
        """
        batch_size = x.shape[0]
        scale_outputs = []
        
        # Process at each scale
        for scale_idx, processor in enumerate(self.scale_processors):
            # Apply scale-specific processing
            scale_output = processor(x)
            
            # Apply attention between scales
            if scale_idx > 0:
                attended_output, _ = self.cross_scale_attention(
                    scale_output,
                    scale_outputs[-1],
                    scale_outputs[-1]
                )
                scale_output = scale_output + attended_output
                
            scale_outputs.append(scale_output)
            
        # Combine scales using learned weights
        normalized_weights = torch.softmax(self.scale_weights, dim=0)
        output = sum(
            out * w for out, w in zip(scale_outputs, normalized_weights)
        )
        
        if return_scale_outputs:
            return output, scale_outputs
        return output 