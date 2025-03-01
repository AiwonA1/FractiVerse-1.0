import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

class PatternEmergence:
    """Detect genuine pattern emergence"""
    def __init__(self):
        self.emergence_threshold = 0.1
        self.stability_threshold = 0.01
        self.history_length = 50
        
    def detect_patterns(self, field_history):
        """Detect naturally emerged patterns"""
        patterns = {}
        
        # Convert history to tensor
        history = torch.stack(field_history)
        
        # Find stable regions
        stable_regions = self._find_stable_regions(history)
        
        # Extract pattern features
        for region in stable_regions:
            pattern = self._extract_pattern(history, region)
            if self._validate_pattern(pattern):
                patterns[len(patterns)] = pattern
                
        return patterns
        
    def _find_stable_regions(self, history):
        """Find regions where patterns have stabilized"""
        # Calculate field differences
        diffs = torch.diff(history, dim=0)
        stability = torch.mean(torch.abs(diffs), dim=(1,2))
        
        # Find stable periods
        stable_points = stability < self.stability_threshold
        regions = self._group_stable_points(stable_points)
        
        return regions 

    def _extract_pattern(self, history, region):
        """Extract pattern features from stable region"""
        # Get field slice for region
        field_slice = history[region]
        
        # Calculate pattern properties
        features = {
            'temporal_evolution': self._analyze_evolution(field_slice),
            'spatial_structure': self._analyze_structure(field_slice[-1]),
            'stability': self._measure_stability(field_slice),
            'complexity': self._measure_complexity(field_slice)
        }
        
        return features
        
    def _validate_pattern(self, pattern):
        """Validate if pattern represents genuine emergence"""
        # Check stability
        if pattern['stability'] < self.stability_threshold:
            return False
        
        # Check complexity
        if pattern['complexity'] < 0.2:  # Too simple
            return False
        
        # Check temporal consistency
        if not self._check_temporal_consistency(pattern['temporal_evolution']):
            return False
        
        return True
        
    def _group_stable_points(self, stable_points):
        """Group consecutive stable points into regions"""
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