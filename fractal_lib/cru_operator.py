import torch
import torch.nn as nn
from typing import Dict, Optional
import psutil
import time
import numpy as np

class CRUOperator:
    """
    Cognitive Resource Unit (CRU) operator for measuring AI computational
    efficiency and resource utilization.
    """
    
    def __init__(
        self,
        memory_threshold: float = 0.8,
        cpu_threshold: float = 0.9,
        gpu_threshold: Optional[float] = 0.85
    ):
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.gpu_threshold = gpu_threshold
        
        # Initialize resource tracking
        self.resource_usage = {
            'memory': [],
            'cpu': [],
            'gpu': [] if torch.cuda.is_available() else None,
            'time': []
        }
        
    def measure_resource_usage(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        warmup_iterations: int = 3
    ) -> Dict[str, float]:
        """
        Measures computational resource usage during model operation.
        """
        # Warmup runs
        for _ in range(warmup_iterations):
            with torch.no_grad():
                model(input_data)
                
        # Begin measurement
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # GPU memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_gpu = torch.cuda.memory_allocated()
            
        # Measure CPU usage
        start_cpu = psutil.cpu_percent()
        
        # Model execution
        with torch.no_grad():
            output = model(input_data)
            
        # Collect measurements
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        measurements = {
            'execution_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'cpu_usage': end_cpu - start_cpu
        }
        
        if torch.cuda.is_available():
            end_gpu = torch.cuda.memory_allocated()
            measurements['gpu_usage'] = (end_gpu - start_gpu) / 1024 / 1024  # MB
            
        # Update resource tracking
        self._update_resource_tracking(measurements)
        
        return measurements
    
    def compute_efficiency_score(self, measurements: Dict[str, float]) -> float:
        """
        Computes overall efficiency score based on resource usage.
        """
        scores = []
        
        # Memory efficiency
        memory_score = 1.0 - (measurements['memory_usage'] / self.memory_threshold)
        scores.append(max(0, memory_score))
        
        # CPU efficiency
        cpu_score = 1.0 - (measurements['cpu_usage'] / (self.cpu_threshold * 100))
        scores.append(max(0, cpu_score))
        
        # GPU efficiency if available
        if 'gpu_usage' in measurements and self.gpu_threshold:
            gpu_score = 1.0 - (measurements['gpu_usage'] / self.gpu_threshold)
            scores.append(max(0, gpu_score))
            
        # Time efficiency (normalized to baseline)
        time_score = 1.0 / (1.0 + measurements['execution_time'])
        scores.append(time_score)
        
        return np.mean(scores) * 100  # Return percentage
    
    def _update_resource_tracking(self, measurements: Dict[str, float]):
        """
        Updates historical resource usage tracking.
        """
        self.resource_usage['memory'].append(measurements['memory_usage'])
        self.resource_usage['cpu'].append(measurements['cpu_usage'])
        self.resource_usage['time'].append(measurements['execution_time'])
        
        if 'gpu_usage' in measurements:
            self.resource_usage['gpu'].append(measurements['gpu_usage'])
            
    def get_resource_trends(self) -> Dict[str, float]:
        """
        Analyzes trends in resource usage over time.
        """
        trends = {}
        for resource, measurements in self.resource_usage.items():
            if measurements:  # Skip if no measurements
                # Calculate trend using linear regression
                x = np.arange(len(measurements))
                slope = np.polyfit(x, measurements, 1)[0]
                trends[resource] = slope
                
        return trends 