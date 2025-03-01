"""
Memory Pattern Retrieval System
Implements advanced pattern retrieval and association
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from .memory_integration import MemoryPattern

@dataclass
class RetrievalResult:
    """Result of pattern retrieval"""
    pattern_id: str
    pattern: torch.Tensor
    similarity: float
    peff_state: Dict[str, float]
    related_patterns: List[str]
    retrieval_time: float

class MemoryRetrieval:
    """Advanced pattern retrieval system"""
    
    def __init__(self, memory_integration):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory = memory_integration
        
        # Retrieval settings
        self.similarity_threshold = 0.7
        self.max_related_patterns = 5
        self.search_depth = 3
        
        # Cache for fast retrieval
        self.pattern_cache: Dict[str, torch.Tensor] = {}
        self.association_cache: Dict[str, List[str]] = {}
        
        print("\n🔍 Memory Retrieval Initialized")
        
    async def retrieve_pattern(self, query: torch.Tensor, peff_state: Optional[Dict[str, float]] = None) -> RetrievalResult:
        """Retrieve most similar pattern"""
        try:
            start_time = time.time()
            
            # Create query vector
            query_vector = self.memory._create_memory_vector(query)
            
            # Find most similar pattern
            best_match = await self._find_best_match(query_vector)
            if not best_match:
                return None
                
            # Find related patterns
            related = await self._find_related_patterns(best_match.id)
            
            # Update access metrics
            self._update_access_metrics(best_match.id)
            
            # Create result
            result = RetrievalResult(
                pattern_id=best_match.id,
                pattern=best_match.vector,
                similarity=best_match.coherence,
                peff_state=best_match.peff_state,
                related_patterns=related,
                retrieval_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            print(f"Pattern retrieval error: {e}")
            return None
            
    async def retrieve_by_association(self, peff_state: Dict[str, float], top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve patterns by PEFF state association"""
        try:
            results = []
            
            # Calculate PEFF similarity for all patterns
            for pattern in self.memory.patterns.values():
                similarity = self._calculate_peff_similarity(pattern.peff_state, peff_state)
                
                if similarity > self.similarity_threshold:
                    result = RetrievalResult(
                        pattern_id=pattern.id,
                        pattern=pattern.vector,
                        similarity=similarity,
                        peff_state=pattern.peff_state,
                        related_patterns=await self._find_related_patterns(pattern.id),
                        retrieval_time=0
                    )
                    results.append(result)
                    
            # Sort by similarity and return top-k
            results.sort(key=lambda x: x.similarity, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"Association retrieval error: {e}")
            return []
            
    async def _find_best_match(self, query_vector: torch.Tensor) -> Optional[MemoryPattern]:
        """Find best matching pattern"""
        best_similarity = -1
        best_pattern = None
        
        for pattern in self.memory.patterns.values():
            similarity = torch.cosine_similarity(
                query_vector.flatten().unsqueeze(0),
                pattern.vector.flatten().unsqueeze(0)
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_pattern = pattern
                
        return best_pattern if best_similarity > self.similarity_threshold else None
        
    async def _find_related_patterns(self, pattern_id: str, depth: int = 1) -> List[str]:
        """Find related patterns through connection graph"""
        if pattern_id in self.association_cache:
            return self.association_cache[pattern_id]
            
        related = set()
        current_level = {pattern_id}
        
        # Traverse connection graph up to search depth
        for _ in range(depth):
            next_level = set()
            for pid in current_level:
                if pid in self.memory.patterns:
                    pattern = self.memory.patterns[pid]
                    next_level.update(pattern.connections)
                    
            related.update(next_level)
            current_level = next_level
            
        # Sort by coherence
        related_patterns = list(related)
        related_patterns.sort(
            key=lambda pid: self.memory.patterns[pid].coherence if pid in self.memory.patterns else 0,
            reverse=True
        )
        
        # Cache result
        self.association_cache[pattern_id] = related_patterns[:self.max_related_patterns]
        
        return self.association_cache[pattern_id]
        
    def _calculate_peff_similarity(self, state1: Dict[str, float], state2: Dict[str, float]) -> float:
        """Calculate similarity between PEFF states"""
        common_keys = set(state1.keys()) & set(state2.keys())
        if not common_keys:
            return 0.0
            
        similarities = []
        for key in common_keys:
            similarities.append(1 - abs(state1[key] - state2[key]))
            
        return sum(similarities) / len(similarities)
        
    def _update_access_metrics(self, pattern_id: str):
        """Update pattern access metrics"""
        if pattern_id in self.memory.patterns:
            pattern = self.memory.patterns[pattern_id]
            pattern.access_count += 1
            pattern.last_access = time.time()
            
            # Clear old cache entries
            if len(self.pattern_cache) > 1000:  # Max cache size
                oldest = min(self.pattern_cache.keys(), key=lambda k: self.memory.patterns[k].last_access)
                del self.pattern_cache[oldest]
                if oldest in self.association_cache:
                    del self.association_cache[oldest] 