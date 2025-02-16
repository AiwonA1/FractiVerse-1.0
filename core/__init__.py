import sys
import os

# Ensure core modules are found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.fractal_cognition import FractiCognition
from core.memory_manager import MemoryManager
from core.fracti_fpu import FractiProcessingUnit
