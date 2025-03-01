"""
FractiVerse Admin UI
Web-based monitoring and control interface
"""

import asyncio
import tornado.web
import tornado.websocket
import json
from typing import Dict, List, Optional
import time
from dataclasses import asdict
import numpy as np

class UIState:
    """Global UI state"""
    def __init__(self):
        self.connected = False
        self.clients = set()
        self.last_pattern_id = None
        self.system_coherence = 0.0
        self.component_status = {}
        self.cognitive_metrics = {}
        self.network_metrics = {}
        self.blockchain_metrics = {}
        self.treasury_metrics = {}

ui_state = UIState()

class MainHandler(tornado.web.RequestHandler):
    """Main UI page handler"""
    def get(self):
        self.render("templates/admin.html")

class WSHandler(tornado.websocket.WebSocketHandler):
    """WebSocket handler for real-time updates"""
    
    def open(self):
        """Client connected"""
        ui_state.clients.add(self)
        ui_state.connected = True
        self.send_initial_state()
        
    def on_close(self):
        """Client disconnected"""
        ui_state.clients.remove(self)
        if not ui_state.clients:
            ui_state.connected = False
            
    def on_message(self, message):
        """Handle incoming messages"""
        data = json.loads(message)
        if data['type'] == 'request_update':
            self.send_state_update()
            
    def send_initial_state(self):
        """Send initial UI state"""
        self.write_message({
            'type': 'initial_state',
            'data': {
                'component_status': ui_state.component_status,
                'cognitive_metrics': ui_state.cognitive_metrics,
                'network_metrics': ui_state.network_metrics,
                'blockchain_metrics': ui_state.blockchain_metrics,
                'treasury_metrics': ui_state.treasury_metrics
            }
        })
        
    def send_state_update(self):
        """Send state update"""
        self.write_message({
            'type': 'state_update',
            'data': {
                'system_coherence': ui_state.system_coherence,
                'last_pattern_id': ui_state.last_pattern_id,
                'cognitive_metrics': ui_state.cognitive_metrics
            }
        })

def start_admin_server(monitor, port: int = 8080):
    """Start admin UI server"""
    app = tornado.web.Application([
        (r"/", MainHandler),
        (r"/ws", WSHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": "static"}),
    ])
    
    app.listen(port)
    print(f"\n🌐 Admin UI server started on port {port}")
    
    # Start background monitoring
    asyncio.create_task(_monitor_system(monitor))

async def _monitor_system(monitor):
    """Background system monitoring"""
    while True:
        try:
            # Update component status
            ui_state.component_status = monitor.component_status
            
            # Update cognitive metrics
            if monitor.cognitive_events:
                latest = monitor.cognitive_events[-1]
                ui_state.cognitive_metrics = {
                    'peff_coherence': latest.metrics['peff_coherence'],
                    'memory_usage': latest.metrics['memory_usage'],
                    'learning_rate': latest.metrics['learning_rate'],
                    'pattern_count': len(monitor.cognitive_events)
                }
                
            # Update network metrics
            ui_state.network_metrics = {
                'peer_count': monitor.system.network.peer_count(),
                'sync_status': monitor.system.network.sync_status(),
                'bandwidth_usage': monitor.system.network.get_bandwidth_usage()
            }
            
            # Update blockchain metrics
            ui_state.blockchain_metrics = {
                'block_height': monitor.system.chain.get_height(),
                'pattern_count': monitor.system.chain.get_pattern_count(),
                'consensus_status': monitor.system.consensus.get_status()
            }
            
            # Update treasury metrics
            ui_state.treasury_metrics = {
                'total_supply': monitor.system.token.total_supply(),
                'reward_pool': monitor.system.treasury.get_reward_pool(),
                'distribution_rate': monitor.system.treasury.get_distribution_rate()
            }
            
            # Broadcast updates
            _broadcast_updates()
            
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"Monitoring error: {e}")
            await asyncio.sleep(5)

def _broadcast_updates():
    """Broadcast updates to all connected clients"""
    if not ui_state.clients:
        return
        
    update = {
        'type': 'metrics_update',
        'data': {
            'cognitive': ui_state.cognitive_metrics,
            'network': ui_state.network_metrics,
            'blockchain': ui_state.blockchain_metrics,
            'treasury': ui_state.treasury_metrics
        }
    }
    
    for client in ui_state.clients:
        client.write_message(update)

def update_ui(event):
    """Update UI with new cognitive event"""
    ui_state.last_pattern_id = event.pattern_id
    ui_state.system_coherence = event.metrics['peff_coherence']
    
    if ui_state.clients:
        update = {
            'type': 'cognitive_event',
            'data': asdict(event)
        }
        for client in ui_state.clients:
            client.write_message(update)

def get_ui_state():
    """Get current UI state"""
    return {
        'connected': ui_state.connected,
        'last_pattern_id': ui_state.last_pattern_id,
        'system_coherence': ui_state.system_coherence,
        'component_status': ui_state.component_status,
        'cognitive_metrics': ui_state.cognitive_metrics,
        'network_metrics': ui_state.network_metrics,
        'blockchain_metrics': ui_state.blockchain_metrics,
        'treasury_metrics': ui_state.treasury_metrics
    } 