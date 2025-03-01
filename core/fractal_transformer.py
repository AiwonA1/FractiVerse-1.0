import torch
import torch.nn as nn
import torch.nn.functional as F

class FractalTransformer(nn.Module):
    """Fractal-based transformer for pattern processing"""
    def __init__(self, initial_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.dim = initial_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Initialize layers
        self.layers = nn.ModuleList([
            FractalTransformerLayer(initial_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(initial_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class FractalTransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attention = FractalAttention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x 