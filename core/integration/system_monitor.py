"""
FractiVerse System Monitor
Tracks and visualizes system performance and cognitive activities
"""

import asyncio
import torch
import numpy as np
from typing import Dict, List
import time
import logging
from dataclasses import dataclass
import webbrowser

from ..cognition.peff import PEFFSystem
from ..chain.blockchain import FractiChain
from ..network.protocol import FractiNet
from ..treasury.management import FractiTreasury
from ..token.contract import FractiToken

@dataclass
class CognitiveEvent:
    """Cognitive event for logging"""
    timestamp: float
    event_type: str
    pattern_id: str
    metrics: Dict[str, float]
    peff_state: Dict[str, float]

class SystemMonitor:
    """Monitors and logs system activities"""
    
    def __init__(self, fractiverse):
        self.system = fractiverse
        
        # Setup logging
        self.logger = logging.getLogger('FractiVerse')
        self.logger.setLevel(logging.INFO)
        
        # Cognitive event log
        self.cognitive_events: List[CognitiveEvent] = []
        self.max_events = 1000
        
        # Component status
        self.component_status = {
            'FractiCognition': False,
            'FractiChain': False,
            'FractiNet': False,
            'FractiTreasury': False,
            'FractiToken': False
        }
        
        # Admin UI port
        self.admin_port = 8080
        
        print("\n📊 System Monitor Initialized")
        
    async def start_monitoring(self):
        """Start system monitoring"""
        try:
            # Verify component initialization
            await self._verify_components()
            
            # Start monitoring tasks
            asyncio.create_task(self._monitor_cognition())
            asyncio.create_task(self._monitor_blockchain())
            asyncio.create_task(self._monitor_network())
            
            # Launch admin UI
            self._launch_admin_ui()
            
            print("\n🔍 System Monitoring Started")
            
        except Exception as e:
            print(f"Monitoring error: {e}")
            
    async def _verify_components(self):
        """Verify all components are properly initialized"""
        try:
            # Verify FractiCognition
            self.component_status['FractiCognition'] = (
                self.system.peff is not None and
                self.system.learning is not None and
                self.system.memory is not None
            )
            
            # Verify FractiChain
            self.component_status['FractiChain'] = (
                self.system.chain is not None and
                self.system.consensus is not None
            )
            
            # Verify FractiNet
            self.component_status['FractiNet'] = (
                self.system.network is not None and
                self.system.network.is_running()
            )
            
            # Verify FractiTreasury
            self.component_status['FractiTreasury'] = (
                self.system.treasury is not None and
                self.system.treasury.is_initialized()
            )
            
            # Verify FractiToken
            self.component_status['FractiToken'] = (
                self.system.token is not None and
                self.system.token.is_deployed()
            )
            
            # Log initialization status
            self.logger.info("Component Status:")
            for component, status in self.component_status.items():
                self.logger.info(f"{component}: {'✓' if status else '✗'}")
                
            return all(self.component_status.values())
            
        except Exception as e:
            self.logger.error(f"Component verification error: {e}")
            return False
            
    async def _monitor_cognition(self):
        """Monitor cognitive activities"""
        while True:
            try:
                # Get current cognitive state
                peff_state = self.system.peff.get_state()
                memory_state = await self.system.memory.get_state()
                learning_state = self.system.learning.get_state()
                
                # Create cognitive event
                event = CognitiveEvent(
                    timestamp=time.time(),
                    event_type="cognitive_update",
                    pattern_id=memory_state.get('latest_pattern_id'),
                    metrics={
                        'peff_coherence': peff_state['peff_coherence'],
                        'memory_usage': memory_state['memory_usage'],
                        'learning_rate': learning_state['learning_rate']
                    },
                    peff_state=peff_state
                )
                
                # Log event
                self._log_cognitive_event(event)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Cognition monitoring error: {e}")
                await asyncio.sleep(5)
                
    def _log_cognitive_event(self, event: CognitiveEvent):
        """Log cognitive event"""
        try:
            # Add to event log
            self.cognitive_events.append(event)
            
            # Trim log if needed
            if len(self.cognitive_events) > self.max_events:
                self.cognitive_events = self.cognitive_events[-self.max_events:]
                
            # Log to file
            self.logger.info(
                f"Cognitive Event: {event.event_type} | "
                f"Pattern: {event.pattern_id} | "
                f"Coherence: {event.metrics['peff_coherence']:.3f}"
            )
            
            # Update admin UI
            self._update_admin_ui(event)
            
        except Exception as e:
            self.logger.error(f"Event logging error: {e}")
            
    def _launch_admin_ui(self):
        """Launch admin UI in browser"""
        try:
            # Start admin server
            from .admin_ui import start_admin_server
            start_admin_server(self, port=self.admin_port)
            
            # Open browser
            webbrowser.open(f"http://localhost:{self.admin_port}")
            
            self.logger.info(f"Admin UI launched on port {self.admin_port}")
            
        except Exception as e:
            self.logger.error(f"Admin UI launch error: {e}")
            
    def _update_admin_ui(self, event: CognitiveEvent):
        """Update admin UI with new event"""
        try:
            # Send update to UI
            from .admin_ui import update_ui
            update_ui(event)
            
        except Exception as e:
            self.logger.error(f"UI update error: {e}") 