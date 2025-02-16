import sys
import os

# Ensure core modules can be found
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from .fractal_cognition import FractiCognition
from .memory_manager import MemoryManager
from .fracti_fpu import FractiProcessingUnit
