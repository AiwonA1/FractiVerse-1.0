"""FractiVerse 1.0 Core Configuration"""
from pathlib import Path
from typing import Dict, Any
import json

# Base Directories
BASE_DIR = Path(__file__).parent.parent
CORE_DIR = BASE_DIR / "core"
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
VIZ_DIR = BASE_DIR / "visualizations"

# Ensure directories exist
for dir_path in [DATA_DIR, LOG_DIR, VIZ_DIR]:
    dir_path.mkdir(exist_ok=True)

# Component Configuration
FRACTIVERSE_CONFIG = {
    "version": "1.0.0",
    "components": {
        "fractal_intelligence": {
            "enabled": True,
            "unipixel_processing": True,
            "recursive_learning": True,
            "vector_dimension": 3
        },
        "fractichain": {
            "enabled": True,
            "network": "testnet",
            "consensus": "recursive_proof_of_intelligence",
            "block_time": 60
        },
        "fractinet": {
            "enabled": True,
            "p2p_port": 9000,
            "max_peers": 100,
            "sync_interval": 30
        },
        "fracti_treasury": {
            "enabled": True,
            "token_name": "FRACTI",
            "initial_supply": 1000000,
            "mining_reward": 10
        },
        "fracti_navigator": {
            "enabled": True,
            "visualization": "3d",
            "update_interval": 1.0
        }
    },
    "peff": {
        "enabled": True,
        "harmonization_interval": 5,
        "quantum_alignment": True
    },
    "logging": {
        "level": "INFO",
        "rotation": "1 day",
        "retention": "30 days"
    },
    "monitoring": {
        "metrics_port": 9090,
        "prometheus": True,
        "visualization": True
    }
}

def load_config() -> Dict[str, Any]:
    """Load FractiVerse configuration"""
    config_file = CORE_DIR / "fractiverse_config.json"
    
    if config_file.exists():
        with open(config_file, "r") as f:
            return json.load(f)
    
    # Save default config if not exists
    with open(config_file, "w") as f:
        json.dump(FRACTIVERSE_CONFIG, f, indent=2)
    
    return FRACTIVERSE_CONFIG

def get_component_config(component_name: str) -> Dict[str, Any]:
    """Get configuration for a specific component"""
    config = load_config()
    return config["components"].get(component_name, {}) 