import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

class FractalNetwork(nn.Module):
    """
    Implements fractal-based decentralized networking with self-similar
    communication patterns and recursive information exchange.
    """
    
    def __init__(
        self,
        network_dim: int,
        num_nodes: int = 8,
        connection_density: float = 0.7,
        recursion_depth: int = 3
    ):
        super().__init__()
        self.network_dim = network_dim
        self.num_nodes = num_nodes
        
        # Node processors
        self.node_processors = nn.ModuleList([
            self._create_node_processor()
            for _ in range(num_nodes)
        ])
        
        # Connection matrices
        self.connection_matrices = nn.Parameter(
            torch.stack([
                self._initialize_connections(connection_density)
                for _ in range(recursion_depth)
            ])
        )
        
        # Message encoders/decoders
        self.message_encoder = nn.Linear(network_dim, network_dim)
        self.message_decoder = nn.Linear(network_dim, network_dim)
        
    def _create_node_processor(self) -> nn.Module:
        """Creates a node processing module."""
        return nn.Sequential(
            nn.Linear(self.network_dim, self.network_dim * 2),
            nn.LayerNorm(self.network_dim * 2),
            nn.GELU(),
            nn.Linear(self.network_dim * 2, self.network_dim)
        )
        
    def _initialize_connections(self, density: float) -> torch.Tensor:
        """Initializes fractal connection pattern."""
        connections = torch.rand(self.num_nodes, self.num_nodes)
        connections = (connections < density).float()
        return connections
        
    def forward(
        self,
        node_states: torch.Tensor,
        return_network_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass implementing fractal network communication.
        """
        batch_size = node_states.shape[0]
        network_states = []
        messages = []
        
        current_states = node_states
        
        # Process through network recursively
        for depth, connections in enumerate(self.connection_matrices):
            # Process each node
            processed_states = []
            for i, processor in enumerate(self.node_processors):
                # Process node state
                node_state = processor(current_states[:, i])
                
                # Generate messages
                node_messages = self.message_encoder(node_state)
                
                # Exchange messages based on connections
                received_messages = torch.matmul(
                    connections[i].unsqueeze(0),
                    node_messages.unsqueeze(1)
                )
                
                messages.append(received_messages)
                processed_states.append(
                    self.message_decoder(received_messages.squeeze())
                )
                
            # Update network state
            current_states = torch.stack(processed_states, dim=1)
            network_states.append(current_states)
            
        if return_network_info:
            return current_states, {
                'network_states': network_states,
                'messages': messages,
                'connections': self.connection_matrices.detach().cpu().numpy()
            }
        return current_states
    
    def compute_network_metrics(self, states: List[torch.Tensor]) -> Dict[str, float]:
        """
        Computes network performance metrics.
        """
        metrics = {
            'connectivity': self._measure_connectivity(),
            'message_efficiency': self._measure_message_efficiency(states),
            'network_stability': self._measure_stability(states)
        }
        return metrics
        
    def _measure_connectivity(self) -> float:
        """Measures network connectivity density."""
        return self.connection_matrices.mean().item()
        
    def _measure_message_efficiency(self, states: List[torch.Tensor]) -> float:
        """Measures efficiency of message propagation."""
        state_changes = torch.stack([
            (states[i+1] - states[i]).abs().mean()
            for i in range(len(states)-1)
        ])
        return state_changes.mean().item()
        
    def _measure_stability(self, states: List[torch.Tensor]) -> float:
        """Measures network state stability."""
        state_vars = torch.stack([
            state.var(dim=1).mean()
            for state in states
        ])
        return 1.0 - state_vars.mean().item() 