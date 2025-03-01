"""
FractiTreasury Management System
Handles resource allocation and rewards
"""

from typing import Dict, List
import time
from dataclasses import dataclass

@dataclass
class ResourceAllocation:
    resource_type: str
    amount: float
    recipient: str
    timestamp: float

class FractiTreasury:
    """Treasury management system"""
    
    def __init__(self):
        self.resources: Dict[str, float] = {
            'compute': 1000.0,
            'storage': 1000.0,
            'network': 1000.0
        }
        self.allocations: List[ResourceAllocation] = []
        self.reward_rate = 0.01  # 1% reward rate
        
        print("\n💎 FractiTreasury Initialized")
        
    def allocate_resources(self, resource_type: str, amount: float, recipient: str) -> bool:
        """Allocate resources to recipient"""
        if resource_type not in self.resources:
            return False
            
        if amount > self.resources[resource_type]:
            return False
            
        allocation = ResourceAllocation(
            resource_type=resource_type,
            amount=amount,
            recipient=recipient,
            timestamp=time.time()
        )
        
        self.resources[resource_type] -= amount
        self.allocations.append(allocation)
        
        return True
        
    def calculate_rewards(self, coherence: float) -> Dict[str, float]:
        """Calculate rewards based on coherence"""
        rewards = {}
        
        for allocation in self.allocations:
            reward = allocation.amount * self.reward_rate * coherence
            if allocation.recipient not in rewards:
                rewards[allocation.recipient] = 0
            rewards[allocation.recipient] += reward
            
        return rewards 