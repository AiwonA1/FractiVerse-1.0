"""FractiVerse 1.0 Core Orchestrator"""
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .unipixel_core import UnipixelCore
from .reality_system import RealitySystem
from .peff_system import PeffSystem
from .cognitive_engine import CognitiveEngine

class FractiVerseOrchestrator:
    """Orchestrates all FractiVerse components and their interactions."""
    
    def __init__(self):
        """Initialize the FractiVerse orchestrator."""
        self.components: Dict[str, Any] = {}
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all components."""
        try:
            # Initialize core components
            self.components["unipixel"] = UnipixelCore()
            self.components["reality"] = RealitySystem()
            self.components["peff"] = PeffSystem()
            self.components["cognition"] = CognitiveEngine()
            
            print("FractiVerse components initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize components: {e}")
            raise
            
    async def start(self) -> bool:
        """Start all FractiVerse components.
        
        Returns:
            bool: True if started successfully
        """
        try:
            start_time = datetime.now()
            
            # Start components in order
            for name, component in self.components.items():
                print(f"Starting {name} component...")
                if hasattr(component, 'start'):
                    if asyncio.iscoroutinefunction(component.start):
                        await component.start()
                    else:
                        component.start()
                else:
                    print(f"Warning: {name} component has no start method")
            
            # Log startup metrics
            startup_duration = (datetime.now() - start_time).total_seconds()
            print(f"Startup duration: {startup_duration}s")
            
            # Verify all components are active
            for name, component in self.components.items():
                if hasattr(component, 'is_active') and not component.is_active():
                    raise Exception(f"{name} component failed to start")
            
            print("FractiVerse system started successfully")
            return True
            
        except Exception as e:
            print(f"Failed to start FractiVerse system: {e}")
            return False
            
    async def _start_core_components(self):
        """Start core components."""
        try:
            # Start Unipixel processing
            self.components["unipixel"].start()
            
            # Start Reality system
            self.components["reality"].start()
            
            # Start PEFF system
            self.components["peff"].start()
            
            # Start Cognitive engine
            self.components["cognition"].start()
            
            print("Core components started successfully")
            
        except Exception as e:
            print(f"Failed to start core components: {e}")
            raise
            
    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through the FractiVerse system.
        
        Args:
            input_data: Dictionary containing input data to process
            
        Returns:
            Dict[str, Any]: Processing results
        """
        command_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        print(f"Processing input {command_id}")
        
        try:
            # Input validation
            if not isinstance(input_data, dict):
                raise ValueError("Input must be a dictionary")
            
            if "coordinates" in input_data:
                if not isinstance(input_data["coordinates"], list):
                    raise ValueError("coordinates must be a list")
                
                for coord in input_data["coordinates"]:
                    if not isinstance(coord, dict):
                        raise ValueError("Each coordinate must be a dictionary")
                    if "position" not in coord:
                        raise ValueError("Each coordinate must have a position")
                    if not isinstance(coord["position"], list) or len(coord["position"]) != 3:
                        raise ValueError("Position must be a list of 3 coordinates")
                    if not all(isinstance(x, (int, float)) for x in coord["position"]):
                        raise ValueError("Position coordinates must be numbers")
                    if "value" in coord and not isinstance(coord["value"], (int, float)):
                        raise ValueError("Coordinate value must be a number")
            
            # Process through cognitive engine
            cognitive_result = self.components["cognition"].process_input(input_data)
            if not cognitive_result:
                raise Exception("Cognitive processing failed")
                
            # Process through reality system
            reality_result = self.components["reality"].process(cognitive_result)
            if not reality_result:
                raise Exception("Reality processing failed")
                
            # Process through PEFF system
            peff_result = self.components["peff"].process(reality_result)
            if not peff_result:
                raise Exception("PEFF processing failed")
                
            # Process through unipixel system
            for coord in peff_result.get("coordinates", []):
                x, y, z = coord["position"]
                value = coord.get("value", 1.0)
                self.components["unipixel"].process_point(x, y, z, value)
                
            return {
                "status": "success",
                "command_id": command_id,
                "cognitive_state": cognitive_result.get("cognitive_state", {}),
                "reality_state": reality_result.get("reality_state", {}),
                "peff_state": peff_result
            }
            
        except ValueError as e:
            print(f"Input validation error {command_id}: {e}")
            return {
                "status": "error",
                "command_id": command_id,
                "error": str(e)
            }
        except Exception as e:
            print(f"Error processing input {command_id}: {e}")
            return {
                "status": "error",
                "command_id": command_id,
                "error": str(e)
            }
            
    async def stop(self):
        """Stop all FractiVerse components."""
        try:
            # Stop in reverse order
            for name, component in reversed(list(self.components.items())):
                print(f"Stopping {name} component...")
                if hasattr(component, 'stop'):
                    if asyncio.iscoroutinefunction(component.stop):
                        await component.stop()
                    else:
                        component.stop()
                else:
                    print(f"Warning: {name} component has no stop method")
            
            # Verify all components are inactive
            for name, component in self.components.items():
                if hasattr(component, 'is_active') and component.is_active():
                    print(f"Warning: {name} component failed to stop")
            
            print("FractiVerse system stopped successfully")
            
        except Exception as e:
            print(f"Failed to stop FractiVerse system: {e}")
            raise

    async def process_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a command through the FractiVerse system.
        
        Args:
            command_data: Dictionary containing command data to process
            
        Returns:
            Dict[str, Any]: Command processing results
        """
        command_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        print(f"Processing command {command_id}")
        
        try:
            command = command_data.get("command")
            if not command:
                raise ValueError("Missing command")
                
            # Process through cognitive engine first
            cognitive_result = self.components["cognition"].process_command(command)
            if not cognitive_result:
                raise Exception("Command processing failed")
                
            # Execute command based on cognitive analysis
            result = {
                "status": "success",
                "command_id": command_id,
                "command": command,
                "cognitive_result": cognitive_result
            }
            
            # Add component-specific results if available
            for component_name, component in self.components.items():
                if hasattr(component, "get_metrics"):
                    result[f"{component_name}_metrics"] = component.get_metrics()
            
            return result
            
        except Exception as e:
            print(f"Error processing command {command_id}: {e}")
            return {
                "status": "error",
                "command_id": command_id,
                "error": str(e)
            } 