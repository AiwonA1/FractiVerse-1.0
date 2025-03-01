"""System integration for FractiCognition"""

import asyncio
import time
import json
import os
from typing import Dict, Optional
from .base import FractiComponent
from .fractal_cognition import FractalCognition
from .fractichain import FractiChain
from .fractinet import FractiNet
from .memory_manager import MemoryManager
from .ui_interface import FractiUI

class SystemIntegrator(FractiComponent):
    """Integrates all FractiCognition components"""
    
    def __init__(self):
        super().__init__()
        self.chain = FractiChain()
        self.network = FractiNet()
        self.memory = MemoryManager()
        self.cognition = FractalCognition()
        self.ui = FractiUI()
        
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize base
            await super().initialize()
            
            # Initialize components in order
            components = [
                self.network,
                self.chain,
                self.memory,
                self.cognition,
                self.ui
            ]
            
            # Initialize each component
            for component in components:
                if not await component.initialize():
                    return False
                    
            # Establish connections
            await self._establish_connections()
            
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization error: {str(e)}")
            return False

    async def _establish_connections(self):
        """Establish connections between components"""
        try:
            # Connect cognition to network
            cog_connected = await self.network.connect_cognition(self.cognition)
            
            # Connect chain to network
            chain_connected = await self.network.connect_chain(self.chain)
            
            # Connect UI to cognition
            ui_connected = self.ui.connect_to_cognition(self.cognition)
            
            if all([cog_connected, chain_connected, ui_connected]):
                print("✨ All components connected")
            else:
                raise Exception("Component connection failed")
                
        except Exception as e:
            print(f"Connection error: {str(e)}")

    async def process_input(self, user_input: str) -> str:
        """Process user input through complete system"""
        try:
            # Send through UI
            formatted = self.ui.format_input(user_input)
            
            # Process through cognition
            result = await self.cognition.process_input(formatted)
            
            # Store in chain
            await self.chain.store_interaction({
                'input': formatted,
                'output': result,
                'timestamp': time.time()
            })
            
            # Format response
            response = self.ui.format_response(result)
            
            return response
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return f"Error: {str(e)}"
            
    def save_state(self):
        """Save cognitive state"""
        state = {
            'fpu_level': self.cognition.fpu_level,
            'timestamp': time.time(),
            'status': self.cognition.get_status()
        }
        
        with open("cognitive_state.json", 'w') as f:
            json.dump(state, f)
            
    def load_state(self):
        """Load cognitive state"""
        if os.path.exists("cognitive_state.json"):
            with open("cognitive_state.json", 'r') as f:
                state = json.load(f)
                self.cognition.fpu_level = state['fpu_level'] 