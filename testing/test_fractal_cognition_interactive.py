"""
FractalCognition Interactive Test Console
---------------------------------------
Provides a web-based interface for interacting with FractiCody's cognitive system.
"""

import os
import sys
import time
import uvicorn
import webbrowser
from datetime import datetime
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect
import asyncio
from core.fractal_cognition import FractalCognition
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.fracticody_engine import FractiCodyEngine

app = FastAPI()
engine = FractiCodyEngine()

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>FractalCognition Interactive Console</title>
    <style>
        body { 
            font-family: 'Arial', sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #f0f2f5; 
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        #output {
            height: 400px;
            overflow-y: auto;
            padding: 10px;
            background: #1e1e1e;
            color: #fff;
            border-radius: 5px;
            margin-bottom: 20px;
            font-family: monospace;
        }
        #input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .status {
            background: #e3f2fd;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .system { color: #4CAF50; }
        .user { color: #2196F3; }
        .error { color: #f44336; }
        .learning { color: #ff9800; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 FractalCognition 1.0 Interactive Console</h1>
        <div class="status" id="status">
            Initializing system...
        </div>
        <div id="output"></div>
        <input type="text" id="input" placeholder="Enter command (type 'help' for commands)"
               onkeypress="if(event.keyCode==13) sendMessage();">
    </div>

    <script>
        let ws = new WebSocket("ws://localhost:8000/ws");
        let output = document.getElementById("output");
        let status = document.getElementById("status");
        
        ws.onmessage = function(event) {
            let data = JSON.parse(event.data);
            if (data.type === 'status') {
                status.innerHTML = data.content;
            } else {
                let div = document.createElement("div");
                div.className = data.type;
                div.textContent = data.content;
                output.appendChild(div);
                output.scrollTop = output.scrollHeight;
            }
        };
        
        function sendMessage() {
            let input = document.getElementById("input");
            if (input.value) {
                ws.send(input.value);
                let div = document.createElement("div");
                div.className = "user";
                div.textContent = "> " + input.value;
                output.appendChild(div);
                input.value = "";
                output.scrollTop = output.scrollHeight;
            }
        }
        
        ws.onopen = function() {
            output.innerHTML += '<div class="system">Connected to FractalCognition system</div>';
        };
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Send initial connection status
        await websocket.send_json({
            "type": "status",
            "content": "🧠 FractiCognition 1.0 Initializing..."
        })
        
        # Initialize base cognitive state
        engine.fractal_cognition._initialize_fractal_patterns()
        
        await websocket.send_json({
            "type": "status",
            "content": f"🧠 FractiCognition 1.0 Active\n" + \
                      f"Base FPU Level: {engine.fractal_cognition.fpu_level * 100:.1f}\n" + \
                      "Ready for interaction..."
        })
        
        while True:
            try:
                user_input = await websocket.receive_text()
                if not user_input:
                    continue
                
                # Check for expansion command
                if "expand" in user_input.lower() and "cognitive" in user_input.lower():
                    await websocket.send_json({
                        "type": "system",
                        "content": "🚀 Beginning cognitive expansion sequence..."
                    })
                    
                    # Start expansion
                    response = engine.fractal_cognition.expand_cognitive_capacity(target_fpu=1000)
                    
                    await websocket.send_json({
                        "type": "system",
                        "content": response
                    })
                    continue
                
                # Normal input processing
                response = engine.fractal_cognition.process_input(user_input)
                
                # Get current metrics
                fpu_level = engine.fractal_cognition.fpu_level * 100
                pattern_quality = engine.fractal_cognition.fpu_metrics['pattern_depth'] * 100
                
                await websocket.send_json({
                    "type": "system",
                    "content": f"Processing Results:\n" + \
                              f"{response}\n\n" + \
                              f"Current Status:\n" + \
                              f"FPU Level: {fpu_level:.1f}\n" + \
                              f"Pattern Quality: {pattern_quality:.1f}%"
                })
                
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except Exception as e:
                print(f"Processing error: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "content": "Processing error - retrying..."
                })
                
    except Exception as e:
        print(f"WebSocket error: {str(e)}")

def find_available_port(start_port=8000, max_attempts=5):
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise RuntimeError("No available ports found")

def run_test():
    """Start the interactive test console"""
    try:
        PORT = find_available_port()
        print(f"Using port: {PORT}")
        
        # Update WebSocket URL in HTML template
        global HTML_CONTENT
        HTML_CONTENT = HTML_CONTENT.replace(
            '"ws://localhost:8000/ws"',
            f'"ws://localhost:{PORT}/ws"'
        )
        
        # Create URL for browser
        url = f'http://localhost:{PORT}'
        print(f"Opening browser at: {url}")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(1.5)  # Wait for server to be ready
            webbrowser.open(url)
        
        import threading
        threading.Thread(target=open_browser).start()
        
        # Start server
        uvicorn.run(app, host="0.0.0.0", port=PORT)
        
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

class FractiCognitionCLI:
    """Interactive CLI for FractiCognition"""
    
    def __init__(self):
        self.cognitive_system = FractalCognition()
        self.history = []
        self.prompt = "🤖 > "
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_welcome(self):
        self.clear_screen()
        print("🌟 Welcome to FractiCognition Interactive CLI 🌟")
        print("===============================================")
        print("Current FPU Level:", f"{self.cognitive_system.fpu_level*100:.2f}%")
        print("Active Patterns:", len(self.cognitive_system.pattern_network.patterns))
        print("System Status: Ready")
        print("\nType 'help' for commands or 'exit' to quit")
        print("===============================================\n")
        
    def print_response(self, response, is_error=False):
        """Print formatted response"""
        if is_error:
            print("\n❌ Error:", response)
        else:
            # Format response blocks
            if isinstance(response, dict):
                print("\n🔹 Response:")
                for key, value in response.items():
                    print(f"  {key}: {value}")
            else:
                print("\n🔹", response)
        print()
        
    async def handle_command(self, command):
        """Handle user command"""
        try:
            if command.lower() == 'exit':
                return False
                
            elif command.lower() == 'help':
                print("\n📚 Available Commands:")
                print("  - expand <n>: Expand cognitive capacity to n FPU")
                print("  - status: Show detailed system status")
                print("  - patterns: Show active patterns")
                print("  - clear: Clear screen")
                print("  - save: Save current state")
                print("  - load: Load saved state")
                print("  - exit: Exit the program")
                print("  Any other input will be processed as a cognitive query\n")
                
            elif command.lower() == 'status':
                status = self.cognitive_system.get_status()
                print("\n📊 System Status:")
                print(f"  FPU Level: {status['fpu_level']*100:.2f}%")
                print(f"  Active Patterns: {status['active_patterns']}")
                print(f"  Pattern Complexity: {status.get('complexity', 0):.4f}")
                print(f"  Learning Efficiency: {status.get('learning', 0):.4f}")
                print(f"  Memory Integration: {status.get('memory', 0):.4f}\n")
                
            elif command.lower().startswith('expand'):
                target = float(command.split()[1])
                print(f"\n🚀 Expanding to {target} FPU...")
                async for status in self.cognitive_system.expand_cognitive_capacity(target):
                    self.clear_screen()
                    print(f"📈 Progress: {status['fpu_level']*100:.2f}%")
                    print(f"Active Patterns: {status.get('patterns', 0)}")
                    print(f"Resonance Events: {status.get('resonance', 0)}")
                    await asyncio.sleep(0.1)
                print("\n✅ Expansion complete!\n")
                
            elif command.lower() == 'patterns':
                patterns = self.cognitive_system.pattern_network.patterns
                print("\n🔮 Active Patterns:")
                for pid, pattern in patterns.items():
                    print(f"\nPattern {pid}:")
                    print(f"  Strength: {pattern['strength']:.2f}")
                    print(f"  Coherence: {pattern['features'].get('coherence', 0):.2f}")
                    print(f"  Connections: {len(pattern['connections'])}")
                print()
                
            elif command.lower() == 'clear':
                self.clear_screen()
                self.print_welcome()
                
            elif command.lower() == 'save':
                self.cognitive_system.save_state()
                print("\n💾 State saved successfully!\n")
                
            elif command.lower() == 'load':
                self.cognitive_system.load_state()
                print("\n📂 State loaded successfully!\n")
                
            else:
                # Process as cognitive query
                response = await self.cognitive_system.process_command(command)
                self.print_response(response)
                
            return True
            
        except Exception as e:
            self.print_response(str(e), is_error=True)
            return True
            
    async def run(self):
        """Run interactive CLI"""
        self.print_welcome()
        
        while True:
            try:
                command = input(self.prompt).strip()
                if not command:
                    continue
                    
                should_continue = await self.handle_command(command)
                if not should_continue:
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
                
            except Exception as e:
                self.print_response(str(e), is_error=True)

if __name__ == "__main__":
    cli = FractiCognitionCLI()
    asyncio.run(cli.run())
