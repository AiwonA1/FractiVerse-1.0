"""
FractiChain 1.0
Fractal blockchain for cognitive pattern storage and validation
"""

import time
import hashlib
import json
from typing import Dict, List, Optional
import torch
from dataclasses import dataclass

@dataclass
class PatternBlock:
    """Block containing cognitive patterns"""
    index: int
    timestamp: float
    patterns: Dict[str, torch.Tensor]
    previous_hash: str
    coherence: float
    nonce: int
    hash: str = ""

class FractiChain:
    """Fractal blockchain implementation"""
    
    def __init__(self):
        self.chain: List[PatternBlock] = []
        self.pending_patterns: Dict[str, torch.Tensor] = {}
        self.min_coherence = 0.7
        
        # Create genesis block
        self._create_genesis_block()
        
        print("\n⛓️ FractiChain 1.0 Initialized")
        
    def _create_genesis_block(self):
        """Create the first block in the chain"""
        genesis = PatternBlock(
            index=0,
            timestamp=time.time(),
            patterns={},
            previous_hash="0",
            coherence=1.0,
            nonce=0
        )
        genesis.hash = self._calculate_block_hash(genesis)
        self.chain.append(genesis)
        
    def _calculate_block_hash(self, block: PatternBlock) -> str:
        """Calculate hash for a block"""
        # Convert patterns to bytes for hashing
        pattern_bytes = b''
        for pattern_id, tensor in block.patterns.items():
            pattern_bytes += pattern_id.encode() + tensor.cpu().numpy().tobytes()
            
        block_content = f"{block.index}{block.timestamp}{pattern_bytes.hex()}{block.previous_hash}{block.coherence}{block.nonce}"
        return hashlib.sha256(block_content.encode()).hexdigest()
        
    def add_pattern(self, pattern_id: str, pattern: torch.Tensor, coherence: float):
        """Add pattern to pending patterns"""
        if coherence >= self.min_coherence:
            self.pending_patterns[pattern_id] = pattern
            
            # Create new block if enough patterns
            if len(self.pending_patterns) >= 10:  # Block size threshold
                self._create_new_block()
                
    def _create_new_block(self) -> Optional[PatternBlock]:
        """Create a new block from pending patterns"""
        if not self.pending_patterns:
            return None
            
        previous_block = self.chain[-1]
        
        new_block = PatternBlock(
            index=len(self.chain),
            timestamp=time.time(),
            patterns=self.pending_patterns.copy(),
            previous_hash=previous_block.hash,
            coherence=self._calculate_block_coherence(),
            nonce=0
        )
        
        # Mine block
        new_block = self._mine_block(new_block)
        
        # Add to chain and clear pending
        self.chain.append(new_block)
        self.pending_patterns.clear()
        
        return new_block
        
    def _calculate_block_coherence(self) -> float:
        """Calculate overall coherence of pending patterns"""
        if not self.pending_patterns:
            return 0.0
            
        coherences = []
        for pattern in self.pending_patterns.values():
            # Calculate pattern coherence
            norm = torch.norm(pattern)
            if norm > 0:
                coherence = torch.sum(torch.abs(pattern)) / (norm * pattern.numel())
                coherences.append(coherence.item())
                
        return sum(coherences) / len(coherences) if coherences else 0.0
        
    def _mine_block(self, block: PatternBlock, difficulty: int = 4) -> PatternBlock:
        """Mine block by finding nonce that produces hash with leading zeros"""
        target = "0" * difficulty
        
        while True:
            block.hash = self._calculate_block_hash(block)
            if block.hash.startswith(target):
                break
            block.nonce += 1
            
        return block
        
    def validate_chain(self) -> bool:
        """Validate entire blockchain"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Verify hash
            if current.previous_hash != previous.hash:
                return False
                
            # Verify block hash
            if current.hash != self._calculate_block_hash(current):
                return False
                
            # Verify coherence
            if current.coherence < self.min_coherence:
                return False
                
        return True
        
    def get_pattern_history(self, pattern_id: str) -> List[torch.Tensor]:
        """Get historical versions of a pattern"""
        history = []
        for block in self.chain:
            if pattern_id in block.patterns:
                history.append(block.patterns[pattern_id])
        return history 