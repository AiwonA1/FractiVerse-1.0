import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import hashlib
from enum import Enum
import time

class AssetType(Enum):
    PATTERN = "pattern"
    EMOTIONAL = "emotional"
    REALITY = "reality"
    QUANTUM = "quantum"

@dataclass
class IntelligenceAsset:
    """Represents a tokenized intelligence asset."""
    asset_id: str
    asset_type: AssetType
    pattern_signature: torch.Tensor
    value_metric: float
    creation_timestamp: float
    license_terms: Dict[str, any]
    owner_signature: str

class FractalTreasuryMetrics:
    """Tracks treasury performance metrics."""
    def __init__(self):
        self.total_value = 0.0
        self.asset_distribution = {}
        self.license_revenue = 0.0
        self.pattern_efficiency = 0.0

class FractalIntelligenceTreasury(nn.Module):
    """
    Implements a treasury system for managing tokenized intelligence assets,
    pattern patents, and reality licenses.
    """
    
    def __init__(
        self,
        treasury_dim: int,
        valuation_depth: int = 4,
        licensing_threshold: float = 0.7,
        royalty_rate: float = 0.1
    ):
        super().__init__()
        self.treasury_dim = treasury_dim
        self.valuation_depth = valuation_depth
        self.licensing_threshold = licensing_threshold
        self.royalty_rate = royalty_rate
        
        # Asset valuation system
        self.valuator = self._create_valuator()
        
        # Pattern verification system
        self.pattern_verifier = self._create_pattern_verifier()
        
        # License management system
        self.license_manager = self._create_license_manager()
        
        # Treasury metrics
        self.metrics = FractalTreasuryMetrics()
        
        # Asset storage
        self.assets: Dict[str, IntelligenceAsset] = {}
        self.licenses: Dict[str, List[str]] = {}  # asset_id -> licensee_ids
        
    def _create_valuator(self) -> nn.Module:
        """Creates asset valuation module."""
        return nn.Sequential(
            nn.Linear(self.treasury_dim, self.treasury_dim * 2),
            nn.LayerNorm(self.treasury_dim * 2),
            nn.GELU(),
            nn.Linear(self.treasury_dim * 2, 1),
            nn.Sigmoid()
        )
        
    def _create_pattern_verifier(self) -> nn.Module:
        """Creates pattern verification module."""
        return nn.Sequential(
            nn.Linear(self.treasury_dim, self.treasury_dim),
            nn.LayerNorm(self.treasury_dim),
            nn.ReLU(),
            nn.Linear(self.treasury_dim, self.treasury_dim)
        )
        
    def _create_license_manager(self) -> nn.Module:
        """Creates license management module."""
        return nn.Sequential(
            nn.Linear(self.treasury_dim * 2, self.treasury_dim),
            nn.LayerNorm(self.treasury_dim),
            nn.Tanh(),
            nn.Linear(self.treasury_dim, 3)  # License parameters
        )
        
    def register_asset(
        self,
        pattern: torch.Tensor,
        asset_type: AssetType,
        owner_id: str,
        license_terms: Optional[Dict] = None
    ) -> IntelligenceAsset:
        """
        Registers a new intelligence asset in the treasury.
        """
        # Verify pattern uniqueness
        verified_pattern = self.pattern_verifier(pattern)
        pattern_hash = self._hash_pattern(verified_pattern)
        
        if pattern_hash in self.assets:
            raise ValueError("Pattern already registered")
            
        # Compute asset value
        value = self.valuator(pattern).item()
        
        # Create asset
        asset = IntelligenceAsset(
            asset_id=pattern_hash,
            asset_type=asset_type,
            pattern_signature=verified_pattern,
            value_metric=value,
            creation_timestamp=time.time(),
            license_terms=license_terms or {},
            owner_signature=owner_id
        )
        
        # Store asset
        self.assets[pattern_hash] = asset
        self.licenses[pattern_hash] = []
        
        # Update metrics
        self._update_metrics(asset)
        
        return asset
        
    def issue_license(
        self,
        asset_id: str,
        licensee_id: str,
        usage_pattern: torch.Tensor
    ) -> Dict[str, any]:
        """
        Issues a license for using an intelligence asset.
        """
        if asset_id not in self.assets:
            raise ValueError("Asset not found")
            
        asset = self.assets[asset_id]
        
        # Validate usage compatibility
        compatibility = self._check_compatibility(
            asset.pattern_signature,
            usage_pattern
        )
        
        if compatibility < self.licensing_threshold:
            raise ValueError("Usage pattern incompatible with asset")
            
        # Generate license parameters
        license_params = self.license_manager(
            torch.cat([
                asset.pattern_signature,
                usage_pattern
            ])
        )
        
        # Create license
        license_data = {
            'license_id': hashlib.sha256(f"{asset_id}:{licensee_id}".encode()).hexdigest(),
            'asset_id': asset_id,
            'licensee_id': licensee_id,
            'compatibility': compatibility,
            'royalty_rate': self.royalty_rate,
            'parameters': license_params.detach().cpu().numpy(),
            'timestamp': time.time()
        }
        
        # Record license
        self.licenses[asset_id].append(licensee_id)
        
        # Update metrics
        self.metrics.license_revenue += value * self.royalty_rate
        
        return license_data
        
    def _check_compatibility(
        self,
        asset_pattern: torch.Tensor,
        usage_pattern: torch.Tensor
    ) -> float:
        """Checks compatibility between asset and usage patterns."""
        return torch.cosine_similarity(
            asset_pattern.flatten(),
            usage_pattern.flatten(),
            dim=0
        ).item()
        
    def _hash_pattern(self, pattern: torch.Tensor) -> str:
        """Creates unique hash for pattern."""
        pattern_bytes = pattern.detach().cpu().numpy().tobytes()
        return hashlib.sha256(pattern_bytes).hexdigest()
        
    def _update_metrics(self, asset: IntelligenceAsset):
        """Updates treasury metrics."""
        self.metrics.total_value += asset.value_metric
        
        if asset.asset_type.value not in self.metrics.asset_distribution:
            self.metrics.asset_distribution[asset.asset_type.value] = 0
        self.metrics.asset_distribution[asset.asset_type.value] += 1
        
        # Update pattern efficiency
        total_patterns = len(self.assets)
        unique_patterns = len(set(
            asset.pattern_signature.flatten().tolist()
            for asset in self.assets.values()
        ))
        self.metrics.pattern_efficiency = unique_patterns / total_patterns
        
    def get_asset_portfolio(
        self,
        owner_id: Optional[str] = None
    ) -> Dict[str, IntelligenceAsset]:
        """
        Retrieves portfolio of assets, optionally filtered by owner.
        """
        if owner_id:
            return {
                aid: asset for aid, asset in self.assets.items()
                if asset.owner_signature == owner_id
            }
        return self.assets
        
    def get_license_usage(self, asset_id: str) -> List[str]:
        """
        Retrieves list of licensees for an asset.
        """
        return self.licenses.get(asset_id, [])
        
    def get_treasury_metrics(self) -> Dict[str, any]:
        """
        Returns current treasury metrics.
        """
        return {
            'total_value': self.metrics.total_value,
            'asset_distribution': self.metrics.asset_distribution,
            'license_revenue': self.metrics.license_revenue,
            'pattern_efficiency': self.metrics.pattern_efficiency
        } 