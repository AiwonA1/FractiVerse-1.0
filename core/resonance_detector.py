import torch
import numpy as np
from scipy.signal import find_peaks

class ResonanceDetector:
    """Detect genuine pattern resonance"""
    def __init__(self):
        self.resonance_threshold = 0.01
        self.history_length = 50
        self.resonance_history = []
        
    def detect_resonance(self, field_history):
        """Find resonating patterns in field evolution"""
        resonance = []
        
        # Convert history to tensor
        history = torch.stack(field_history)
        
        # Find stable regions
        stable_regions = self._find_stable_regions(history)
        
        # Extract resonance patterns
        for region in stable_regions:
            pattern = self._extract_resonance(history, region)
            if self._validate_resonance(pattern):
                resonance.append(pattern)
                
        return resonance
        
    def _find_stable_regions(self, history):
        """Find regions of stability in evolution"""
        # Calculate field differences
        diffs = torch.diff(history, dim=0)
        stability = torch.mean(torch.abs(diffs), dim=(1,2))
        
        # Find stable periods
        stable = stability < self.resonance_threshold
        regions = self._group_stable_points(stable)
        
        return regions 

    def _extract_resonance(self, history, region):
        """Extract resonance pattern from stable region"""
        # Get field slice
        field_slice = history[region]
        
        # Calculate resonance properties
        features = {
            'temporal_evolution': self._analyze_evolution(field_slice),
            'spatial_structure': self._analyze_structure(field_slice[-1]),
            'stability': self._measure_stability(field_slice),
            'complexity': self._measure_complexity(field_slice[-1])
        }
        
        return features
        
    def _validate_resonance(self, pattern):
        """Validate resonance pattern"""
        # Check stability
        if pattern['stability'] < self.resonance_threshold:
            return False
        
        # Check complexity
        if pattern['complexity'] < 0.2:
            return False
        
        # Check temporal consistency
        if not self._check_temporal_consistency(pattern['temporal_evolution']):
            return False
        
        return True
        
    def _group_stable_points(self, stable_points):
        """Group consecutive stable points"""
        regions = []
        current_region = []
        
        for i, stable in enumerate(stable_points):
            if stable:
                current_region.append(i)
            elif current_region:
                if len(current_region) >= 5:  # Minimum region size
                    regions.append(current_region)
                current_region = []
                
        if current_region and len(current_region) >= 5:
            regions.append(current_region)
        
        return regions

    def _analyze_evolution(self, field_slice):
        """Analyze temporal evolution"""
        # Calculate temporal gradients
        gradients = torch.gradient(field_slice, dim=0)
        
        # Measure evolution characteristics
        features = {
            'speed': torch.mean(torch.abs(gradients[0])),
            'acceleration': torch.mean(torch.diff(gradients[0], dim=0)),
            'consistency': torch.std(gradients[0])
        }
        
        return features

    def _analyze_structure(self, field):
        """Analyze spatial structure"""
        # Calculate spatial gradients
        gradients = torch.gradient(field)
        
        # Measure structural properties
        features = {
            'gradient_magnitude': torch.mean(torch.abs(gradients[0])),
            'gradient_direction': torch.mean(torch.atan2(gradients[1], gradients[0])),
            'symmetry': self._measure_symmetry(field)
        }
        
        return features

    def _measure_stability(self, field_slice):
        """Measure stability of field evolution"""
        # Calculate temporal gradients
        gradients = torch.gradient(field_slice, dim=0)
        
        # Measure stability metrics
        stability = {
            'mean_change': torch.mean(torch.abs(gradients[0])),
            'variance': torch.var(gradients[0]),
            'max_change': torch.max(torch.abs(gradients[0]))
        }
        
        return 1.0 - min(1.0, stability['mean_change'])

    def _measure_complexity(self, field):
        """Measure pattern complexity"""
        # Spatial gradients
        gradients = torch.gradient(field)
        grad_magnitude = torch.sqrt(gradients[0]**2 + gradients[1]**2)
        
        # Complexity metrics
        complexity = {
            'gradient_entropy': self._calculate_entropy(grad_magnitude),
            'pattern_density': torch.mean(torch.abs(field)),
            'spatial_variance': torch.var(field)
        }
        
        return (complexity['gradient_entropy'] + 
                complexity['pattern_density'] + 
                complexity['spatial_variance']) / 3.0

    def _calculate_entropy(self, tensor):
        """Calculate entropy of tensor values"""
        # Normalize to probabilities
        probs = torch.abs(tensor) / torch.sum(torch.abs(tensor))
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
        return entropy.item()

    def _check_temporal_consistency(self, evolution):
        """Check temporal consistency of pattern evolution"""
        # Check evolution metrics
        speed_consistent = evolution['speed'] < 0.5
        acceleration_stable = evolution['acceleration'] < 0.1
        pattern_stable = evolution['consistency'] < 0.2
        
        return speed_consistent and acceleration_stable and pattern_stable

    def _measure_symmetry(self, field):
        """Measure pattern symmetry"""
        # Check horizontal symmetry
        h_sym = torch.mean(torch.abs(field - torch.flip(field, [1])))
        # Check vertical symmetry
        v_sym = torch.mean(torch.abs(field - torch.flip(field, [0])))
        # Check rotational symmetry
        r_sym = torch.mean(torch.abs(field - torch.rot90(field, 2, [0,1])))
        
        return 1.0 - min(1.0, (h_sym + v_sym + r_sym) / 3.0) 