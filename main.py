"""FractiCognition Main Server"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import webbrowser
from datetime import datetime
import json
from core.system import FractiSystem
from core.knowledge_harvester import KnowledgeHarvester
from core import FractiCodyEngine
import random
import time
from fastapi.templating import Jinja2Templates
import threading
from core.training_coordinator import TrainingCoordinator

# Initialize FastAPI
app = FastAPI(title="FractiCognition")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize system
system = None
harvester = None
browser_launched = False

def launch_browser():
    """Launch browser after server starts"""
    global browser_launched
    if not browser_launched:
        try:
            # Wait a bit for server to be ready
            import time
            time.sleep(2)
            url = "http://127.0.0.1:8000"
            webbrowser.open(url)
            print(f"\n🌐 Launched interface at {url}")
            browser_launched = True
        except Exception as e:
            print(f"Browser launch error: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global system
    try:
        print("\n🚀 Starting FractiCody system...")
        
        # Initialize system
        system = FractiSystem()
        await system.initialize()
        
        # Initialize and start training
        trainer = TrainingCoordinator(system)
        asyncio.create_task(trainer.start_training())
        
        # Start monitoring
        asyncio.create_task(continuous_monitoring(system))
        
        # Launch browser in a separate thread
        threading.Thread(target=launch_browser).start()
        
        print("\n✅ System startup complete")
        
    except Exception as e:
        print(f"❌ Startup failed: {str(e)}")
        raise

async def continuous_monitoring(engine):
    """Monitor system metrics and health"""
    while True:
        try:
            metrics = {
                name: component.get_metrics() 
                for name, component in engine.components.items()
            }
            
            print("\n📊 System Status:")
            for name, component_metrics in metrics.items():
                print(f"├── {name}: {component_metrics}")
            
            await asyncio.sleep(10)
            
        except Exception as e:
            print(f"Monitoring error: {e}")
            await asyncio.sleep(10)

@app.get("/")
async def get():
    """Serve web interface"""
    try:
        with open("interface/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        print(f"Failed to load interface: {e}")
        return HTMLResponse(content=f"<h1>Failed to load interface</h1><p>Error: {str(e)}</p>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        print("⚡ WebSocket client connected")
        
        # Send initial state
        initial_state = {
            'type': 'initial_state',
            'data': {
                'component_status': system.get_component_status(),
                'cognitive_metrics': system.get_cognitive_metrics(),
                'patterns': system.memory.get_recent_patterns(),
                'learning_log': system.memory.get_learning_log()
            }
        }
        await websocket.send_json(initial_state)
        
        # Start pattern monitoring task
        pattern_task = asyncio.create_task(monitor_patterns(websocket))
        
        while True:
            try:
                data = await websocket.receive_json()
                print(f"📥 Received: {data}")
                
                if data.get('type') == 'request_update':
                    state_update = {
                        'type': 'state_update',
                        'data': {
                            'cognitive_metrics': system.get_cognitive_metrics(),
                            'patterns': system.memory.get_recent_patterns(),
                            'learning_log': system.memory.get_learning_log()
                        }
                    }
                    await websocket.send_json(state_update)
                    print("📤 Sent state update")
                
            except WebSocketDisconnect:
                pattern_task.cancel()
                break
                
    except Exception as e:
        print(f"❌ WebSocket error: {str(e)}")
    finally:
        try:
            await websocket.close()
        except:
            pass

async def monitor_patterns(websocket: WebSocket):
    """Monitor and send pattern updates"""
    while True:
        try:
            # Get new patterns
            new_patterns = await system.memory.get_new_patterns()
            if new_patterns:
                await websocket.send_json({
                    'type': 'pattern_update',
                    'data': {
                        'patterns': new_patterns,
                        'timestamp': time.time()
                    }
                })
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Pattern monitoring error: {e}")
            await asyncio.sleep(1)

def start_server():
    """Start the server"""
    try:
        print("\n🧠 FractiCognition 1.0")
        
        # Create required directories
        os.makedirs('static', exist_ok=True)
        os.makedirs('memory', exist_ok=True)
        os.makedirs('interface', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Try different ports if 8000 is in use
        ports = [8000, 8001, 8002, 8003]
        server_started = False
        
        for port in ports:
            try:
                print(f"\n🔄 Attempting to start server on port {port}")
                
                # Check if port is in use
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                
                if result == 0:
                    print(f"❌ Port {port} is in use, trying next port...")
                    continue
                    
                # Start server
                import uvicorn
                config = uvicorn.Config(
                    "main:app",
                    host="127.0.0.1",
                    port=port,
                    reload=True,
                    log_level="info",
                    reload_delay=1.0
                )
                server = uvicorn.Server(config)
                server_started = True
                print(f"✅ Server starting on http://127.0.0.1:{port}")
                server.run()
                break
                
            except Exception as e:
                print(f"❌ Error on port {port}: {str(e)}")
                continue
                
        if not server_started:
            print("\n❌ Failed to start server on any port")
            print("💡 Try manually killing processes:")
            print("   1. Run: lsof -i :8000")
            print("   2. Note the PID")
            print("   3. Run: kill -9 <PID>")
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Server error: {str(e)}")

def generate_test_data():
    """Generate test data for UI demo"""
    return {
        'fpu_level': random.uniform(0.1, 1.0),
        'pattern_count': random.randint(100, 1000),
        'coherence': random.uniform(0.5, 1.0),
        'learning_rate': random.uniform(0.001, 0.1)
    }

if __name__ == "__main__":
    start_server()
