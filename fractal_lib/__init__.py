"""
FractiVerse 1.0 - Fractal, Unipixel, & Cognitive Processing Extensions Library

A comprehensive library for fractal-based AI processing, unipixel cognition,
and recursive blockchain operations.
"""

# Core Components
from .unipixel_core import UnipixelCore, UnipixelState
from .peff_system import PEFFSystem, FractalLayer, PEFFNode
from .fractal_neuron import FractalNeuron
from .recursive_layer import RecursiveLayer
from .multi_scale_operator import MultiScaleOperator
from .unipixel_operator import UnipixelOperator

# Advanced Processing
from .quantum_entanglement import QuantumEntangler
from .deep_recursive_memory import DeepRecursiveMemory
from .fractal_harmonizing import FractalHarmonizer

# Blockchain & Treasury
from .blockchain_fractal import FractalBlockchain
from .fractal_tokenomics import FractalTokenomics
from .fractal_treasury import FractalTreasury
from .fractal_intelligence_treasury import (
    FractalIntelligenceTreasury,
    IntelligenceAsset,
    AssetType,
    FractalTreasuryMetrics
)

# Reality Systems
from .fractal_storytelling import FractalStoryteller
from .aivfiar_forge import AIVFIARForge, RealityBlueprint, RealityLayer

# Network & Interface
from .fractinet_system import FractiNet, NetworkNode, NodeType
from .fractichain_ledger import FractiChainLedger, MemoryConstellation, FractiChainBlock
from .unipixel_interface import UnipixelInterface, ViewMode, ViewportState

__version__ = '1.0.0'
__author__ = 'FractiVerse Team'

# Package metadata
__all__ = [
    # Core
    'UnipixelCore', 'UnipixelState',
    'PEFFSystem', 'FractalLayer', 'PEFFNode',
    'FractalNeuron', 'RecursiveLayer',
    'MultiScaleOperator', 'UnipixelOperator',
    
    # Advanced Processing
    'QuantumEntangler',
    'DeepRecursiveMemory',
    'FractalHarmonizer',
    
    # Blockchain & Treasury
    'FractalBlockchain',
    'FractalTokenomics',
    'FractalTreasury',
    'FractalIntelligenceTreasury',
    'IntelligenceAsset',
    'AssetType',
    'FractalTreasuryMetrics',
    
    # Reality Systems
    'FractalStoryteller',
    'AIVFIARForge',
    'RealityBlueprint',
    'RealityLayer',
    
    # Network & Interface
    'FractiNet',
    'NetworkNode',
    'NodeType',
    'FractiChainLedger',
    'MemoryConstellation',
    'FractiChainBlock',
    'UnipixelInterface',
    'ViewMode',
    'ViewportState'
] 