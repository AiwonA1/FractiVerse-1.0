import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np

class UnipixelOperator(nn.Module):
    """
    Implements unipixel-driven AI cognition through self-replicating knowledge units.
    Enables intelligence compression and hypermagnification of cognitive patterns.
    """
    
    def __init__(
        self,
        dimension: int,
        compression_ratio: float = 0.5,
        replication_factor: int = 2,
        hypermagnification_enabled: bool = True
    ):
        super().__init__()
        self.dimension = dimension
        self.compression_ratio = compression_ratio
        self.replication_factor = replication_factor
        
        compressed_dim = int(dimension * compression_ratio)
        
        # Compression layers
        self.compressor = nn.Sequential(
            nn.Linear(dimension, compressed_dim),
            nn.LayerNorm(compressed_dim),
            nn.ReLU()
        )
        
        # Knowledge replication module
        self.replicator = nn.ModuleList([
            nn.Linear(compressed_dim, compressed_dim)
            for _ in range(replication_factor)
        ])
        
        # Hypermagnification module
        if hypermagnification_enabled:
            self.magnifier = nn.Sequential(
                nn.Linear(compressed_dim, dimension),
                nn.LayerNorm(dimension),
                nn.Sigmoid()
            )
            
        # Knowledge coherence gate
        self.coherence_gate = nn.Parameter(torch.ones(compressed_dim))
        
    def forward(
        self, 
        x: torch.Tensor,
        return_compressed: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass implementing unipixel compression and replication.
        """
        # Compress input to unipixel representation
        compressed = self.compressor(x)
        compressed = compressed * torch.sigmoid(self.coherence_gate)
        
        # Replicate knowledge through self-similar patterns
        replicated_states = []
        for replicator in self.replicator:
            replicated = replicator(compressed)
            replicated_states.append(replicated)
            
        # Combine replicated knowledge
        unified_knowledge = torch.stack(replicated_states).mean(0)
        
        # Apply hypermagnification if enabled
        if hasattr(self, 'magnifier'):
            output = self.magnifier(unified_knowledge)
        else:
            output = unified_knowledge
            
        if return_compressed:
            return output, compressed
        return output
    
    def compute_compression_efficiency(self, x: torch.Tensor) -> float:
        """
        Calculates the efficiency of the unipixel compression.
        """
        original_entropy = self._calculate_entropy(x)
        compressed = self.compressor(x)
        compressed_entropy = self._calculate_entropy(compressed)
        
        return compressed_entropy / original_entropy
        
    def _calculate_entropy(self, x: torch.Tensor) -> float:
        """
        Calculates the information entropy of a tensor.
        """
        # Convert to probabilities using softmax
        probs = torch.softmax(x.flatten(), dim=0).detach().numpy()
        # Remove zeros to avoid log(0)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs)) 