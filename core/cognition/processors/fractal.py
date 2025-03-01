"""
Fractal Pattern Processors
Implements self-similarity engine and recursive template system
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
import torch.nn.functional as F

class FractalProcessor:
    """Implements fractal pattern processing and analysis"""
    
    def __init__(self, dimensions: Tuple[int, int] = (256, 256)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dimensions = dimensions
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Initialize template system
        self.templates = self._initialize_templates()
        
        print("\n🌀 Fractal Processor Initialized")
        
    def _initialize_templates(self) -> Dict[str, torch.Tensor]:
        """Initialize fractal template system"""
        templates = {
            'mandelbrot': self._generate_mandelbrot(),
            'julia': self._generate_julia(),
            'sierpinski': self._generate_sierpinski(),
            'koch': self._generate_koch()
        }
        return templates
        
    def _generate_mandelbrot(self, max_iter: int = 100) -> torch.Tensor:
        """Generate Mandelbrot set template"""
        x = torch.linspace(-2, 1, self.dimensions[0]).to(self.device)
        y = torch.linspace(-1.5, 1.5, self.dimensions[1]).to(self.device)
        c = x.unsqueeze(1) + 1j * y.unsqueeze(0)
        
        z = torch.zeros_like(c)
        mask = torch.ones_like(c, dtype=torch.bool)
        
        for _ in range(max_iter):
            z[mask] = z[mask]**2 + c[mask]
            mask = torch.abs(z) <= 2
            
        return mask.float()
        
    def analyze_self_similarity(self, pattern: torch.Tensor, scales: List[float] = None) -> float:
        """Analyze pattern self-similarity across scales"""
        if scales is None:
            scales = [self.phi**i for i in range(-2, 3)]
            
        similarities = []
        base_pattern = pattern.unsqueeze(0).unsqueeze(0)
        
        for scale in scales:
            scaled_size = (
                int(self.dimensions[0] * scale),
                int(self.dimensions[1] * scale)
            )
            scaled = F.interpolate(base_pattern, size=scaled_size, mode='bilinear')
            scaled = F.interpolate(scaled, size=self.dimensions, mode='bilinear')
            
            similarity = F.cosine_similarity(
                base_pattern.flatten(1),
                scaled.flatten(1)
            ).item()
            similarities.append(similarity)
            
        return np.mean(similarities)
        
    def extract_fractal_features(self, pattern: torch.Tensor) -> Dict[str, float]:
        """Extract fractal features from pattern"""
        features = {
            'self_similarity': self.analyze_self_similarity(pattern),
            'fractal_dimension': self._calculate_fractal_dimension(pattern),
            'template_correlations': self._calculate_template_correlations(pattern)
        }
        return features
        
    def _calculate_fractal_dimension(self, pattern: torch.Tensor) -> float:
        """Calculate fractal dimension using box-counting method"""
        scales = [2**i for i in range(1, 5)]
        counts = []
        
        for scale in scales:
            boxes = pattern.unfold(0, scale, scale).unfold(1, scale, scale)
            count = torch.sum(torch.sum(boxes, dim=-1) > 0)
            counts.append(count.item())
            
        # Calculate dimension from log-log plot
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        return -coeffs[0]  # Negative slope gives dimension 