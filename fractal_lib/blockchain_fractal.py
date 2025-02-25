import torch
import torch.nn as nn
from typing import Dict, List, Optional
import hashlib
import time

class FractalBlockchain:
    """
    Implements fractal-based blockchain verification and self-similar 
    transaction processing.
    """
    
    def __init__(
        self,
        fractal_dimension: float = 1.6,
        verification_depth: int = 3,
        security_threshold: float = 0.85
    ):
        self.fractal_dimension = fractal_dimension
        self.verification_depth = verification_depth
        self.security_threshold = security_threshold
        
        self.chain = []
        self.pending_transactions = []
        
    def create_block(
        self,
        transactions: List[Dict],
        previous_hash: str
    ) -> Dict:
        """Creates a new fractal block with self-similar verification."""
        block = {
            'timestamp': time.time(),
            'transactions': transactions,
            'previous_hash': previous_hash,
            'fractal_patterns': self._generate_fractal_patterns(transactions),
            'nonce': 0
        }
        
        # Proof of fractal work
        while not self._verify_fractal_proof(block):
            block['nonce'] += 1
            
        return block
    
    def _generate_fractal_patterns(
        self,
        transactions: List[Dict]
    ) -> List[torch.Tensor]:
        """Generates self-similar patterns for transaction verification."""
        patterns = []
        for depth in range(self.verification_depth):
            pattern = self._compute_fractal_hash(
                transactions,
                scale=2 ** depth
            )
            patterns.append(pattern)
        return patterns
    
    def _compute_fractal_hash(
        self,
        data: List[Dict],
        scale: int
    ) -> torch.Tensor:
        """Computes a fractal hash at a specific scale."""
        serialized = str(sorted(data, key=lambda x: str(x))).encode()
        raw_hash = hashlib.sha256(serialized).hexdigest()
        
        # Convert hash to tensor and reshape with fractal dimension
        hash_tensor = torch.tensor([int(raw_hash[i:i+2], 16) 
                                  for i in range(0, 64, 2)])
        return hash_tensor.reshape(scale, -1)
    
    def _verify_fractal_proof(self, block: Dict) -> bool:
        """Verifies the fractal proof of work."""
        patterns = block['fractal_patterns']
        
        # Check self-similarity across scales
        similarity_scores = []
        for i in range(len(patterns) - 1):
            score = self._compute_pattern_similarity(
                patterns[i],
                patterns[i + 1]
            )
            similarity_scores.append(score)
            
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        return avg_similarity > self.security_threshold
    
    def _compute_pattern_similarity(
        self,
        pattern1: torch.Tensor,
        pattern2: torch.Tensor
    ) -> float:
        """Computes similarity between fractal patterns."""
        # Resize patterns to same dimension
        if pattern1.shape != pattern2.shape:
            pattern2 = torch.nn.functional.interpolate(
                pattern2.unsqueeze(0).unsqueeze(0),
                size=pattern1.shape,
                mode='bilinear'
            ).squeeze()
            
        return torch.cosine_similarity(
            pattern1.flatten(),
            pattern2.flatten(),
            dim=0
        ).item() 