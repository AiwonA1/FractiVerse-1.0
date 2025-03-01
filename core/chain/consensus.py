"""
FractiChain Consensus System
Implements pattern-based consensus mechanism
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

@dataclass
class ConsensusVote:
    node_id: str
    block_hash: str
    coherence_score: float
    timestamp: float

class FractalConsensus:
    """Pattern-based consensus mechanism"""
    
    def __init__(self):
        self.min_votes = 3  # Minimum votes needed
        self.vote_timeout = 30  # Seconds
        self.pending_votes: Dict[str, List[ConsensusVote]] = {}
        self.validated_blocks: List[str] = []
        
        print("\n🔍 Fractal Consensus Initialized")
        
    def submit_vote(self, node_id: str, block_hash: str, coherence: float):
        """Submit a validation vote for a block"""
        vote = ConsensusVote(
            node_id=node_id,
            block_hash=block_hash,
            coherence_score=coherence,
            timestamp=time.time()
        )
        
        if block_hash not in self.pending_votes:
            self.pending_votes[block_hash] = []
            
        self.pending_votes[block_hash].append(vote)
        
        # Check if consensus reached
        return self.check_consensus(block_hash)
        
    def check_consensus(self, block_hash: str) -> bool:
        """Check if consensus reached for block"""
        if block_hash not in self.pending_votes:
            return False
            
        votes = self.pending_votes[block_hash]
        
        # Remove expired votes
        current_time = time.time()
        votes = [v for v in votes if current_time - v.timestamp <= self.vote_timeout]
        
        if len(votes) < self.min_votes:
            return False
            
        # Calculate average coherence
        avg_coherence = sum(v.coherence_score for v in votes) / len(votes)
        
        # Require 70% coherence and minimum votes
        if avg_coherence >= 0.7:
            self.validated_blocks.append(block_hash)
            return True
            
        return False 