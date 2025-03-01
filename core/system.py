from core.components.memory import MemoryManager
from core.components.cognition import FractalCognition
from core.components.network import FractiNet
import asyncio
import os
import json
from typing import Dict

class FractiSystem:
    def __init__(self):
        """Initialize system components"""
        # Initialize components first
        self.memory = None
        self.cognition = None
        self.network = None
        
        # Then set them up
        self._setup_components()

    def _setup_components(self):
        """Setup and link system components"""
        try:
            # Create required directories first
            os.makedirs('memory', exist_ok=True)
            self._initialize_memory_files()

            # Initialize components in order
            self.memory = MemoryManager()
            self.cognition = FractalCognition()
            self.network = FractiNet()
            
            # Link components after all are initialized
            if self.memory and self.cognition:
                self.cognition.memory = self.memory
                self.memory.cognitive_metrics = self.cognition.cognitive_metrics
            
            print("✅ System components initialized")
        except Exception as e:
            print(f"❌ Component setup error: {e}")
            raise  # Re-raise to prevent partial initialization

    def _initialize_memory_files(self):
        """Initialize memory storage files"""
        memory_files = {
            'memory/patterns.json': {},
            'memory/relationships.json': {}
        }
        
        for file_path, default_content in memory_files.items():
            try:
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        json.dump(default_content, f)
            except Exception as e:
                print(f"Failed to initialize {file_path}: {e}")
                raise  # Re-raise to prevent partial initialization

    async def initialize(self):
        """Initialize all system components"""
        if not all([self.memory, self.cognition, self.network]):
            raise RuntimeError("System components not properly initialized")

        print("\n🚀 Initializing FractiCognition 1.0...")
        
        # Initialize components in order
        await self.memory.initialize()
        await self.cognition.initialize()
        await self.network.initialize()
        
        # Start continuous learning process
        asyncio.create_task(self._continuous_learning())
        
        print("✨ All components initialized")
        return True
        
    async def _continuous_learning(self):
        """Handle continuous learning and cognitive growth"""
        while True:
            try:
                # Process active learning cycle
                await self.memory._memory_growth()
                
                # Update cognitive metrics
                self.cognition.cognitive_metrics['fpu_level'] *= 1.05  # Increase FPU
                self.memory.cognitive_metrics = self.cognition.cognitive_metrics
                
                # Log system status
                print("\n📊 System Status:")
                print(f"├── FPU Level: {self.cognition.cognitive_metrics['fpu_level']*100:.2f}%")
                print(f"├── Learning Stage: {self.memory.memory_metrics['learning_stage']}")
                print(f"└── Active Patterns: {len(self.memory.memory_metrics['active_patterns'])}")
                
                await asyncio.sleep(15)
            except Exception as e:
                print(f"Learning cycle error: {e}")
                await asyncio.sleep(15)

    async def process(self, input_data: str) -> Dict:
        """Process input through the system"""
        try:
            # Create initial pattern
            pattern = {
                'content': input_data,
                'type': 'user_input',
                'cognitive_level': self.cognition.cognitive_metrics['fpu_level']
            }
            
            # Learn pattern
            learning_result = await self.memory.learn_pattern(pattern)
            
            # Process through cognition
            response = await self.cognition.process({'content': input_data})
            
            return {
                'content': response['content'],
                'metrics': {
                    'memory': self.memory.memory_metrics,
                    'cognitive': self.cognition.cognitive_metrics
                }
            }
        except Exception as e:
            print(f"Processing error: {e}")
            return {'content': 'Processing error occurred', 'metrics': {}}

    def get_component_status(self) -> dict:
        return {
            'cognition': self.cognition.is_active(),
            'blockchain': self.chain.is_active(),
            'network': self.network.is_active(),
            'treasury': self.treasury.is_active()
        }
        
    def get_cognitive_metrics(self) -> dict:
        return {
            'fpu_level': self.cognition.get_fpu_level(),
            'pattern_count': len(self.memory.patterns),
            'coherence': self.cognition.get_coherence(),
            'learning_rate': self.cognition.get_learning_rate()
        }
        
    def get_network_metrics(self) -> dict:
        return {
            'connected_nodes': len(self.network.nodes),
            'active_connections': len(self.network.active_connections),
            'packets_processed': self.network.get_packet_count(),
            'bandwidth_usage': self.network.get_bandwidth_usage()
        }
        
    def get_blockchain_metrics(self) -> dict:
        return {
            'block_height': len(self.chain.chain),
            'transaction_count': self.chain.get_transaction_count(),
            'mining_difficulty': self.chain.get_difficulty(),
            'hash_rate': self.chain.get_hash_rate()
        } 