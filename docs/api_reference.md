# FractiVerse API Reference

## Core Components

### UnipixelCore
```python
class UnipixelCore(nn.Module):
    def __init__(
        self,
        dimension: int,
        recursion_depth: int = 4,
        peff_alignment: float = 0.8,
        reality_channels: List[str] = ["LinearVerse", "FractiVerse", "AIVFIAR"]
    )
```

### PEFFSystem
```python
class PEFFSystem(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_layers: int = 5,
        harmony_threshold: float = 0.8,
        coherence_factor: float = 0.7
    )
```

### AIVFIARForge
```python
class AIVFIARForge(nn.Module):
    def __init__(
        self,
        reality_dim: int,
        num_layers: int = 5,
        stability_threshold: float = 0.8,
        quantum_coupling: float = 0.7
    )
```

## Network Components

### FractiNet
```python
class FractiNet(nn.Module):
    def __init__(
        self,
        network_dim: int,
        num_nodes: int = 16,
        echo_radius: float = 0.8,
        bandwidth_capacity: float = 1.0
    )
```

### FractiChainLedger
```python
class FractiChainLedger(nn.Module):
    def __init__(
        self,
        memory_dim: int,
        constellation_size: int = 64,
        memory_compression: float = 0.5,
        coherence_threshold: float = 0.7
    )
```

## Treasury Components

### FractalIntelligenceTreasury
```python
class FractalIntelligenceTreasury(nn.Module):
    def __init__(
        self,
        treasury_dim: int,
        valuation_depth: int = 4,
        licensing_threshold: float = 0.7,
        royalty_rate: float = 0.1
    )
``` 