"""FractiCody Core Package"""

# Version and configuration
VERSION = "1.0.0"

# Base components
from .base import FractiComponent

# Import components
from .fracticody_engine import FractiCodyEngine
from .memory_manager import MemoryManager
from .metrics_manager import MetricsManager
from .fractal_cognition import FractalCognition
from .fracti_decision_engine import FractiDecisionEngine
from .fracti_fpu import FractiProcessingUnit

# Export components
__all__ = [
    'FractiComponent',
    'FractiCodyEngine',
    'MemoryManager',
    'MetricsManager',
    'FractalCognition',
    'FractiDecisionEngine',
    'FractiProcessingUnit'
]
