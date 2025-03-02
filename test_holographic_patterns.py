import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from core.fractichain import FractiChain, FractiBlock
from core.memory_manager import MemoryManager
import asyncio
import torch
import numpy as np
import time
import json
from typing import Dict, List, Optional
import webbrowser
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import uvicorn
import threading
import traceback
import sys
from core.quantum.hologram import FractalVector3D, QuantumHologram
from core.metrics_manager import MetricsManager
import socket
import random

class HolographicMemoryDisplay:
    """Display template for holographic memories"""
    
    @staticmethod
    def format_memory(pattern: 'HolographicPattern', include_quantum: bool = False) -> Dict:
        """Format memory for display"""
        display = {
            "content": {
                "main": pattern.content,
                "type": "holographic_pattern",
                "created": time.strftime('%Y-%m-%d %H:%M:%S', 
                                      time.localtime(pattern.timestamp))
            },
            "location": {
                "x": float(pattern.vector[0].real),
                "y": float(pattern.vector[1].real),
                "z": float(pattern.vector[2].real),
                "coherence": float(torch.abs(pattern.vector).mean())
            },
            "connections": {
                "count": len(pattern.connections),
                "nearest": [
                    {
                        "content": conn.content,
                        "distance": float(torch.dist(pattern.vector, conn.vector))
                    }
                    for conn in pattern.connections[:3]  # Show top 3 connections
                ]
            },
            "metadata": {
                "access_count": pattern.access_count,
                "last_accessed": pattern.last_accessed,
                "importance": pattern.importance
            }
        }
        
        if include_quantum:
            display["quantum_state"] = {
                "magnitude": float(torch.abs(pattern.vector).sum()),
                "phase": float(torch.angle(pattern.vector).mean()),
                "coherence": float(torch.abs(pattern.vector).std())
            }
            
        return display

class HolographicPattern:
    def __init__(self, content: str, vector: torch.Tensor = None):
        self.content = content
        self.vector = vector if vector is not None else self._generate_vector()
        self.timestamp = time.time()
        self.connections = []
        self.access_count = 0
        self.last_accessed = None
        self.importance = 0.5
        
    def _generate_vector(self) -> torch.Tensor:
        """Generate 3D holographic vector"""
        return torch.randn(3, dtype=torch.complex64)
        
    def move(self, direction: torch.Tensor):
        """Move pattern in 3D space"""
        self.vector += direction
        
    def access(self):
        """Record pattern access"""
        self.access_count += 1
        self.last_accessed = time.time()
        
    def connect(self, other: 'HolographicPattern'):
        """Create connection to another pattern"""
        self.connections.append(other)
        
    def __repr__(self):
        return f"HoloPattern({self.content}, pos={self.vector.real.tolist()})"

    @classmethod
    def from_dict(cls, data: Dict) -> 'HolographicPattern':
        """Create pattern from stored data"""
        pattern = cls(
            content=data["content"],
            vector=torch.tensor(data["vector"], dtype=torch.complex64)
        )
        pattern.timestamp = data.get("timestamp", time.time())
        if "connections" in data:
            pattern.connections = [
                cls.from_dict(conn_data) 
                for conn_data in data["connections"]
            ]
        return pattern

class MockMetricsManager:
    """Mock metrics manager for testing"""
    def __init__(self):
        self.initialized = False
        self.metrics = {
            'block_count': 0,
            'chain_health': 1.0,
            'quantum_coherence': 0.95
        }
    
    async def initialize(self) -> bool:
        self.initialized = True
        return True
        
    def record_metric(self, name: str, value: float):
        self.metrics[name] = value
        logger.debug(f"Recorded metric {name}: {value}")
        
    def get_metrics(self) -> dict:
        return self.metrics
        
    def is_active(self) -> bool:
        return self.initialized

async def setup_chain():
    """Setup chain with dependencies"""
    try:
        # Create chain first
        chain = FractiChain()
        logger.debug("FractiChain instance created")
        
        # Create and initialize metrics manager
        metrics_manager = MockMetricsManager()
        success = await metrics_manager.initialize()
        if not success:
            logger.error("Failed to initialize metrics manager")
            return None
        logger.debug("Metrics manager initialized")
        
        # Inject dependency before chain initialization
        chain.inject_dependency('metrics_manager', metrics_manager)
        logger.debug("Metrics manager injected into chain")
        
        # Now initialize chain
        success = await chain.initialize()
        if not success:
            logger.error("Chain initialization failed")
            return None
            
        logger.debug("Chain setup completed successfully")
        return chain
        
    except Exception as e:
        logger.error(f"Chain setup failed: {e}")
        traceback.print_exc()
        return None

async def test_fractichain_operations():
    print("\n🧠 Testing FractiChain Holographic Operations")
    
    try:
        # Initialize chain
        chain = await setup_chain()
        if not chain:
            raise Exception("Failed to initialize chain")
            
        # Create test patterns
        patterns = [
            HolographicPattern("Quantum Entanglement Theory", 
                             vector=torch.tensor([0.1, 0.2, 0.3], dtype=torch.complex64)),
            HolographicPattern("Neural Network Architecture", 
                             vector=torch.tensor([0.4, 0.5, 0.6], dtype=torch.complex64)),
            HolographicPattern("Consciousness Framework", 
                             vector=torch.tensor([0.7, 0.8, 0.9], dtype=torch.complex64))
        ]
        logger.debug(f"Created {len(patterns)} test patterns")
        
        # Create connections
        patterns[0].connect(patterns[1])
        patterns[1].connect(patterns[2])
        patterns[2].connect(patterns[0])
        logger.debug("Pattern connections created")
        
        # Store in chain
        for pattern in patterns:
            await store_pattern(chain, pattern)
            print(f"✅ Stored: {pattern}")
            print(f"   Position: {pattern.vector.real.tolist()}")
            print(f"   Connections: {len(pattern.connections)}")
        
        # Add this after pattern creation
        logger.info("Verifying memory storage...")
        for pattern in patterns:
            logger.info(f"Pattern: {pattern.content}")
            logger.info(f"  Position: {pattern.vector.real.tolist()}")
            logger.info(f"  Coherence: {torch.abs(pattern.vector).mean():.3f}")
            logger.info(f"  Connections: {len(pattern.connections)}")
            
            # Verify pattern is in memory
            memory_id = chain.generate_memory_id(pattern.content)
            stored_pattern = chain.get_pattern(memory_id)
            if stored_pattern:
                logger.info("  ✅ Pattern verified in memory")
            else:
                logger.error("  ❌ Pattern not found in memory")
        
        # Test 2: Pattern Movement
        print("\n🔄 Testing Pattern Movement...")
        pattern = patterns[0]
        old_pos = pattern.vector.clone()
        move_vector = torch.tensor([0.1, 0.1, 0.1], dtype=torch.complex64)
        
        await move_pattern(chain, pattern, move_vector)
        print(f"Moved pattern from {old_pos.real.tolist()} to {pattern.vector.real.tolist()}")
        
        # Test 3: Pattern Relationships
        print("\n🔗 Testing Pattern Relationships...")
        for pattern in patterns:
            connections = await get_pattern_connections(chain, pattern)
            print(f"\nPattern: {pattern.content}")
            print(f"Connected to:")
            for conn in connections:
                print(f"  - {conn.content}")
                print(f"    Position: {conn.vector.real.tolist()}")
        
        # Test 4: Pattern Search
        print("\n🔍 Testing Pattern Search...")
        search_point = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        nearest = await find_nearest_patterns(chain, search_point, k=2)
        print(f"\nNearest patterns to {search_point.real.tolist()}:")
        for pattern, distance in nearest:
            print(f"  - {pattern.content} (distance: {distance:.3f})")
            print(f"    Position: {pattern.vector.real.tolist()}")
        
        # Test 5: Pattern Coherence
        print("\n📊 Testing Pattern Coherence...")
        for pattern in patterns:
            coherence = await calculate_pattern_coherence(chain, pattern)
            print(f"Pattern '{pattern.content}' coherence: {coherence:.3f}")
        
        return chain, patterns
        
    except Exception as e:
        logger.error(f"Test operations failed: {str(e)}")
        traceback.print_exc()
        return None, None

async def store_pattern(chain: FractiChain, pattern: 'HolographicPattern') -> bool:
    """Store pattern in chain"""
    try:
        # Convert pattern to dictionary
        pattern_data = {
            "type": "pattern",
            "content": pattern.content,
            "position": pattern.vector.real.tolist(),
            "coherence": float(torch.abs(pattern.vector).mean()),
            "connections": [p.content for p in pattern.connections],
            "timestamp": time.time()
        }
        
        # Store in chain
        return chain.store_pattern(pattern_data)
        
    except Exception as e:
        logger.error(f"Pattern storage failed: {e}")
        return False

async def move_pattern(chain: FractiChain, pattern: HolographicPattern, 
                      move_vector: torch.Tensor):
    """Move pattern in 3D space"""
    pattern.vector += move_vector
    block = FractiBlock(
        data={
            "type": "pattern_movement",
            "pattern_content": pattern.content,
            "new_vector": pattern.vector.tolist(),
            "movement": move_vector.tolist(),
            "timestamp": time.time()
        },
        previous_hash=chain.chain[-1].hash
    )
    chain.chain.append(block)
    chain.quantum_states[block.hash] = pattern.vector

async def get_pattern_connections(chain: FractiChain, pattern: HolographicPattern):
    """Get pattern connections"""
    connections = []
    for block in chain.chain:
        if block.data.get("type") == "holographic_pattern":
            if block.data.get("content") != pattern.content:
                for conn_data in block.data.get("connections", []):
                    if conn_data["content"] == pattern.content:
                        # Reconstruct connected pattern
                        conn_pattern = HolographicPattern.from_dict({
                            "content": block.data["content"],
                            "vector": block.data["vector"],
                            "timestamp": block.data.get("timestamp", time.time())
                        })
                        connections.append(conn_pattern)
    return connections

async def find_nearest_patterns(chain: FractiChain, point: torch.Tensor, k: int = 3):
    """Find k nearest patterns to a point"""
    distances = []
    for block in chain.chain:
        if block.data.get("type") == "holographic_pattern":
            # Reconstruct pattern from block data
            pattern = HolographicPattern.from_dict(block.data)
            distance = torch.dist(point, pattern.vector)
            distances.append((pattern, float(distance)))
    
    return sorted(distances, key=lambda x: x[1])[:k]

async def calculate_pattern_coherence(chain: FractiChain, pattern: HolographicPattern):
    """Calculate pattern coherence based on connections"""
    if not pattern.connections:
        return 0.0
        
    coherences = []
    for conn in pattern.connections:
        vector_diff = torch.abs(pattern.vector - conn.vector)
        coherence = 1.0 / (1.0 + vector_diff.mean())
        coherences.append(coherence)
        
    return float(torch.tensor(coherences).mean())

# Create FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize components
hologram = QuantumHologram(dimensions=(256, 256, 256))
metrics = MetricsManager()
chain = FractiChain()

# Inject dependencies
chain.inject_dependency('metrics_manager', metrics)

def find_available_port(start_port: int = 61950, end_port: int = 62000) -> int:
    """Find an available port in the given range"""
    for port in range(start_port, end_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                s.listen(1)
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports in range {start_port}-{end_port}")

async def test_holographic_memory():
    """Test holographic memory visualization"""
    try:
        # Initialize components
        hologram = QuantumHologram(dimensions=(256, 256, 256))
        chain = FractiChain()
        
        # Create FastAPI app
        app = FastAPI()
        app.mount("/static", StaticFiles(directory="static"), name="static")
        templates = Jinja2Templates(directory="templates")
        
        # Find available port
        port = find_available_port(start_port=61950, end_port=62000)

        # Add root route handler
        @app.get("/")
        async def root(request: Request):
            return templates.TemplateResponse(
                "admin.html",
                {"request": request, "title": "FractiVerse Navigator"}
            )
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    # Generate test pattern data
                    pattern = {
                        "magnitude": (torch.rand(256, 256) * 0.5 + 0.5).numpy().tolist(),
                        "phase": (torch.rand(256, 256) * 2 * np.pi - np.pi).numpy().tolist(),
                        "coords": {
                            "x": float(np.random.uniform(-2, 2)),
                            "y": float(np.random.uniform(-2, 2)), 
                            "z": float(np.random.uniform(-2, 2))
                        },
                        "coherence": float(np.random.uniform(0.5, 1.0))
                    }
                    
                    # Get system metrics
                    metrics_data = {
                        "fpu_level": float(metrics.metrics.get('fpu_level', 0.001)),  # Access metrics dict
                        "coherence": float(metrics.metrics.get('coherence', 0.8)),  # Default if not set
                        "pattern_count": len(chain.chain),
                        "learning_rate": float(metrics.metrics.get('learning_rate', 0.005))
                    }
                    
                    # Send updates
                    await websocket.send_json({
                        "type": "update",
                        "pattern": {
                            "type": "pattern_update",
                            "id": f"pattern_{int(time.time() * 1000)}",
                            "data": {
                                "magnitude": pattern["magnitude"],
                                "phase": pattern["phase"],
                                "holographic_coords": pattern["coords"],
                                "coherence": pattern["coherence"]
                            }
                        },
                        "metrics": metrics_data
                    })
                    
                    await asyncio.sleep(1)
                    
            except WebSocketDisconnect:
                print("Client disconnected")

        return app, hologram, chain, port

    except Exception as e:
        logger.error(f"Test initialization failed: {e}")
        return None, None, None, None

def find_free_port():
    """Find an available port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

async def store_pattern_in_holographic_memory(pattern, hologram, chain):
    """Store pattern in both quantum hologram and holographic memory"""
    
    logger.info("\n🔮 Processing pattern in quantum hologram...")
    # First process in quantum hologram
    coherence = await hologram.store_pattern(pattern)
    coherence_val = float(coherence) if isinstance(coherence, (int, float)) else 0.0
    
    # Create 3D coordinates with more spread
    coords = {
        "x": float(2.0 * (torch.rand(1) - 0.5)),
        "y": float(2.0 * (torch.rand(1) - 0.5)), 
        "z": float(2.0 * (torch.rand(1) - 0.5))
    }
    
    pattern_id = f"pattern_{int(time.time())}"
    
    # Format pattern data for visualization - match navigator.js exactly
    # Ensure pattern is 256x256 and normalized
    resized_pattern = pattern[:256, :256]
    resized_pattern = resized_pattern / torch.norm(resized_pattern)
    
    # Convert to magnitude and phase
    magnitude = torch.abs(resized_pattern)
    phase = torch.angle(resized_pattern)
    
    # Normalize magnitude to 0-1 range for visualization
    magnitude = magnitude / torch.max(magnitude)
    
    pattern_data = {
        "magnitude": magnitude.flatten().tolist(),
        "phase": phase.flatten().tolist(),
        "holographic_coords": coords,
        "coherence": coherence_val
    }
    
    # Log data for debugging
    logger.debug(f"Pattern data stats:")
    logger.debug(f"Magnitude shape: {len(pattern_data['magnitude'])}")
    logger.debug(f"Phase shape: {len(pattern_data['phase'])}")
    logger.debug(f"Magnitude range: {min(pattern_data['magnitude'])} to {max(pattern_data['magnitude'])}")
    
    # Store in holographic memory
    memory_data = {
        "type": "holographic_pattern",
        "id": pattern_id,
        "timestamp": float(time.time()),
        "content": str({
            "pattern": {
                "real": [float(x) for x in pattern.real.flatten()[:100]],
                "imag": [float(x) for x in pattern.imag.flatten()[:100]]
            },
            "shape": [256, 256]
        }),
        "metadata": {
            "coordinates": coords,
            "stats": {
                "mean": float(torch.mean(magnitude)),
                "std": float(torch.std(magnitude)),
                "max": float(torch.max(magnitude)),
                "min": float(torch.min(magnitude))
            },
            "coherence": coherence_val
        }
    }
    
    try:
        # Store in chain
        success = chain.store_pattern(memory_data)
        
        if success:
            logger.info(f"✅ Pattern {pattern_id} stored at ({coords['x']:.2f}, {coords['y']:.2f}, {coords['z']:.2f})")
        else:
            logger.error(f"❌ Failed to store pattern {pattern_id}")
            
        return pattern_id, coords, coherence_val, success, pattern_data
            
    except Exception as e:
        logger.error(f"❌ Storage error: {str(e)}")
        return pattern_id, coords, coherence_val, False, pattern_data

@app.get("/")
async def get_admin(request: Request):
    """Serve admin interface"""
    try:
        return templates.TemplateResponse(
            "admin.html",
            {
                "request": request,
                "title": "Holographic Memory Viewer"
            }
        )
    except Exception as e:
        logger.error(f"Admin page error: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests"""
    return JSONResponse(content={"status": "ok"})

def launch_browser(port):
    """Launch browser after delay"""
    try:
        time.sleep(2)  # Wait for server to start
        url = f"http://localhost:{port}"
        webbrowser.open(url)
        print(f"✅ Browser launched at {url}")
    except Exception as e:
        logger.error(f"Browser launch failed: {e}")

if __name__ == "__main__":
    try:
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run test
        app, hologram, chain, port = loop.run_until_complete(test_holographic_memory())
        
        if app and port:
            print(f"\n6️⃣ Starting UI server on port {port}...")
            print(f"Open http://localhost:{port} in your browser")
            
            # Start browser in separate thread
            threading.Thread(target=lambda: launch_browser(port), daemon=True).start()
            
            # Start server
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
        else:
            print("\n❌ Test failed to initialize components")
            
    except KeyboardInterrupt:
        print("\n👋 Test terminated by user")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        traceback.print_exc() 