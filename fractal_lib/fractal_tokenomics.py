import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

class FractalTokenomics(nn.Module):
    """
    Implements fractal-based token distribution and economic modeling with
    self-similar incentive structures.
    """
    
    def __init__(
        self,
        total_supply: int,
        num_scales: int = 4,
        distribution_factor: float = 0.7,
        incentive_strength: float = 0.5
    ):
        super().__init__()
        self.total_supply = total_supply
        self.num_scales = num_scales
        
        # Token distribution scales
        self.distribution_scales = nn.Parameter(
            torch.ones(num_scales) * distribution_factor
        )
        
        # Incentive mechanisms
        self.incentive_generators = nn.ModuleList([
            self._create_incentive_generator()
            for _ in range(num_scales)
        ])
        
        # Distribution controller
        self.distribution_controller = nn.Parameter(
            torch.ones(num_scales) * incentive_strength
        )
        
    def _create_incentive_generator(self) -> nn.Module:
        """Creates an incentive generation module."""
        return nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def compute_distribution(
        self,
        activity_metrics: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Computes token distribution based on fractal activity patterns.
        """
        batch_size = activity_metrics.shape[0]
        distributions = []
        incentives = []
        
        # Generate distributions at each scale
        for scale_idx in range(self.num_scales):
            # Generate incentive multiplier
            incentive = self.incentive_generators[scale_idx](
                activity_metrics.unsqueeze(-1)
            )
            incentives.append(incentive)
            
            # Apply scale-specific distribution
            scale_distribution = (
                self.total_supply * 
                self.distribution_scales[scale_idx] * 
                incentive
            )
            distributions.append(scale_distribution)
            
        # Combine distributions with controller weights
        controller_weights = torch.softmax(
            self.distribution_controller,
            dim=0
        )
        
        final_distribution = sum(
            dist * weight 
            for dist, weight in zip(distributions, controller_weights)
        )
        
        if return_details:
            return final_distribution, {
                'scale_distributions': distributions,
                'incentives': incentives,
                'controller_weights': controller_weights.detach().cpu().numpy()
            }
        return final_distribution
    
    def validate_distribution(self, distribution: torch.Tensor) -> bool:
        """
        Validates if distribution adheres to tokenomic constraints.
        """
        total_distributed = distribution.sum()
        return (
            total_distributed <= self.total_supply and
            torch.all(distribution >= 0)
        ) 