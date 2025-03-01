"""
FractiNet Protocol Implementation
Handles node communication and pattern distribution
"""

import asyncio
import json
from typing import Dict, List, Optional
import torch
from dataclasses import dataclass
import time

@dataclass
class NetworkNode:
    id: str
    address: str
    port: int
    last_seen: float
    coherence: float

class FractiNet:
    """Fractal network protocol implementation"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.nodes: Dict[str, NetworkNode] = {}
        self.pattern_buffer: List[torch.Tensor] = []
        self.server = None
        
        print("\n🌐 FractiNet Protocol Initialized")
        
    async def start(self):
        """Start network server"""
        self.server = await asyncio.start_server(
            self._handle_connection, 
            self.host, 
            self.port
        )
        print(f"📡 Network listening on {self.host}:{self.port}")
        
    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming connections"""
        addr = writer.get_extra_info('peername')
        
        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    break
                    
                message = json.loads(data.decode())
                await self._process_message(message, writer)
                
        except Exception as e:
            print(f"Connection error: {e}")
            
        finally:
            writer.close()
            await writer.wait_closed()
            
    async def _process_message(self, message: Dict, writer: asyncio.StreamWriter):
        """Process incoming network message"""
        msg_type = message.get('type')
        
        if msg_type == 'pattern':
            # Handle pattern distribution
            pattern = torch.tensor(message['data'])
            await self._distribute_pattern(pattern)
            
        elif msg_type == 'node_announce':
            # Handle node announcement
            self._register_node(message['node_id'], message['address'], message['port'])
            
    async def _distribute_pattern(self, pattern: torch.Tensor):
        """Distribute pattern to network nodes"""
        message = {
            'type': 'pattern',
            'data': pattern.cpu().numpy().tolist()
        }
        
        data = json.dumps(message).encode()
        
        for node in self.nodes.values():
            try:
                reader, writer = await asyncio.open_connection(node.address, node.port)
                writer.write(data)
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                
            except Exception as e:
                print(f"Distribution error to {node.id}: {e}")
                
    def _register_node(self, node_id: str, address: str, port: int):
        """Register new network node"""
        self.nodes[node_id] = NetworkNode(
            id=node_id,
            address=address,
            port=port,
            last_seen=time.time(),
            coherence=0.0
        ) 