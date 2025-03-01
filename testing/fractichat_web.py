import asyncio
import os
import sys
from datetime import datetime
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
import uvicorn
import socket
from contextlib import closing
import webbrowser
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.fractal_cognition import FractalCognition

def find_free_port():
    """Find a free port to use"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
        return port

def launch_browser(port):
    """Launch browser after short delay to ensure server is up"""
    time.sleep(1.5)  # Wait for server to start
    url = f"http://localhost:{port}"
    webbrowser.open(url)

app = FastAPI()
cognitive_system = FractalCognition()

# Initialize the system
cognitive_system.bootstrap_cognition()
cognitive_system.initialize_core_knowledge()

# Update WebSocket URL in HTML content
port = find_free_port()
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>FractiCognition Neural Interface</title>
    <style>
        body { 
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f0f2f5;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        #chat-window {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background: #1e1e1e;
            color: #fff;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        #neural-status {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        #neural-viz {
            height: 150px;
            background: #fafafa;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 20px;
            padding: 10px;
        }
        #input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }
        .user { 
            background: #1976d2;
            margin-left: 20%;
            color: white;
        }
        .assistant {
            background: #2e7d32;
            margin-right: 20%;
            color: white;
        }
        .neural-activity {
            color: #ff9800;
            font-style: italic;
        }
        .status-bar {
            display: flex;
            justify-content: space-between;
            background: #37474f;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 FractiCognition Neural Interface 1.0</h1>
        
        <div class="status-bar">
            <span>Neural Capacity: <span id="capacity">0.01%</span></span>
            <span>Active Patterns: <span id="patterns">0</span></span>
            <span>Processing State: <span id="state">Ready</span></span>
        </div>

        <div id="neural-status">
            Initializing neural network...
        </div>

        <div id="neural-viz">
            <canvas id="activity-viz" width="960" height="130"></canvas>
        </div>

        <div id="chat-window"></div>

        <input type="text" id="input" placeholder="Chat with FractiCognition (or use /commands: /expand, /status, /clear)">
    </div>

    <script>
        let ws = new WebSocket("ws://localhost:{port}/ws");
        let chatWindow = document.getElementById("chat-window");
        let neuralStatus = document.getElementById("neural-status");
        let canvas = document.getElementById("activity-viz");
        let ctx = canvas.getContext("2d");
        
        // Visualization setup
        let activityData = new Array(100).fill(0);
        
        function updateViz(activity) {
            activityData.push(activity);
            activityData.shift();
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath();
            ctx.moveTo(0, canvas.height - activityData[0] * canvas.height);
            
            for(let i = 1; i < activityData.length; i++) {
                ctx.lineTo(i * (canvas.width/100), canvas.height - activityData[i] * canvas.height);
            }
            
            ctx.strokeStyle = '#4CAF50';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        
        ws.onopen = function() {
            console.log("Connected to FractiCognition");
            chatWindow.innerHTML += '<div class="message system">Connected to FractiCognition Neural Network</div>';
        };
        
        ws.onmessage = function(event) {
            let data = JSON.parse(event.data);
            
            if(data.type === 'neural_activity') {
                updateViz(data.activity_level);
                document.getElementById("capacity").textContent = (data.fpu_level * 100).toFixed(2) + '%';
                document.getElementById("patterns").textContent = data.active_patterns;
                document.getElementById("state").textContent = data.processing_state;
                
                let activityDiv = document.createElement("div");
                activityDiv.className = "message neural-activity";
                activityDiv.textContent = `🔄 ${data.description}`;
                chatWindow.appendChild(activityDiv);
            }
            else if(data.type === 'message') {
                let messageDiv = document.createElement("div");
                messageDiv.className = "message " + data.role;
                messageDiv.textContent = data.content;
                chatWindow.appendChild(messageDiv);
            }
            else if(data.type === 'status') {
                neuralStatus.innerHTML = data.content;
            }
            
            chatWindow.scrollTop = chatWindow.scrollHeight;
        };

        ws.onerror = function(error) {
            console.error("WebSocket Error:", error);
            chatWindow.innerHTML += '<div class="message system error">Connection error</div>';
        };

        ws.onclose = function() {
            console.log("Disconnected from FractiCognition");
            chatWindow.innerHTML += '<div class="message system">Disconnected from neural network</div>';
        };
        
        document.getElementById("input").onkeypress = function(e) {
            if(e.keyCode === 13) {
                let input = this.value;
                if(input.trim()) {
                    ws.send(input);
                    this.value = "";
                }
            }
        };
    </script>
</body>
</html>
"""

@app.on_event("startup")
async def startup_event():
    """Launch browser when server starts"""
    import threading
    threading.Thread(target=launch_browser, args=(port,), daemon=True).start()

@app.get("/")
async def get():
    return HTMLResponse(HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("\n✅ WebSocket connection established")
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "content": f"""
            <h3>🌟 FractiCognition Neural Network Ready</h3>
            <p>Current Neural Capacity: {cognitive_system.fpu_level * 100:.2f}%</p>
            <p>Active Patterns: {len(cognitive_system.pattern_network.patterns)}</p>
            <p>Try these commands:</p>
            <ul>
                <li>/expand - Grow neural capacity</li>
                <li>/status - Show detailed status</li>
                <li>/clear - Clear conversation</li>
            </ul>
            """
        })
        
        while True:
            message = await websocket.receive_text()
            print(f"\n👤 User: {message}")
            
            await websocket.send_json({
                "type": "message",
                "role": "user",
                "content": message
            })
            
            if message.startswith('/'):
                await handle_command(websocket, message[1:].lower())
            else:
                await process_neural_input(websocket, message)
                
    except Exception as e:
        print(f"\n❌ WebSocket error: {e}")
        await websocket.close()

async def handle_command(websocket: WebSocket, command: str):
    """Handle system commands"""
    try:
        if command == 'expand':
            print("\n🧠 Starting neural expansion...")
            async for status in cognitive_system.expand_cognitive_capacity(100):
                progress = f"Neural growth: {status['fpu_level']*100:.1f}%"
                print(f"📈 {progress}")
                
                await websocket.send_json({
                    "type": "neural_activity",
                    "activity_level": status['fpu_level'],
                    "fpu_level": status['fpu_level'],
                    "active_patterns": status.get('patterns', 0),
                    "processing_state": "Expanding",
                    "description": progress
                })
                await asyncio.sleep(0.1)
            print("✨ Neural expansion complete!")
            
        elif command == 'status':
            status = cognitive_system.get_status()
            status_text = f"""
            <h3>Neural Network Status:</h3>
            <p>📊 Capacity: {status['fpu_level']*100:.1f}%</p>
            <p>🧠 Active Patterns: {status['active_patterns']}</p>
            <p>🔄 Pattern Complexity: {status.get('complexity', 0):.3f}</p>
            <p>📈 Learning Rate: {status.get('learning', 0):.3f}</p>
            <p>💫 Memory Integration: {status.get('memory', 0):.3f}</p>
            """
            print(f"\n📊 Status: FPU Level {status['fpu_level']*100:.1f}%, {status['active_patterns']} patterns")
            await websocket.send_json({
                "type": "status",
                "content": status_text
            })
            
        elif command == 'clear':
            await websocket.send_json({
                "type": "status",
                "content": "Conversation cleared"
            })
            print("\n🧹 Conversation cleared")
            
    except Exception as e:
        print(f"\n❌ Command error: {e}")
        await websocket.send_json({
            "type": "message",
            "role": "system",
            "content": f"Error processing command: {str(e)}"
        })

async def process_neural_input(websocket: WebSocket, message: str):
    """Process input through neural network"""
    try:
        print("\n🤔 Processing through neural network...")
        await websocket.send_json({
            "type": "neural_activity",
            "activity_level": cognitive_system.fpu_level,
            "fpu_level": cognitive_system.fpu_level,
            "active_patterns": len(cognitive_system.pattern_network.patterns),
            "processing_state": "Processing",
            "description": "Processing input through neural pathways..."
        })
        
        response = await cognitive_system.process_command(message)
        print(f"🧠 Response: {response}")
        
        await websocket.send_json({
            "type": "message",
            "role": "assistant",
            "content": response
        })
        
    except Exception as e:
        print(f"\n❌ Processing error: {e}")
        await websocket.send_json({
            "type": "message",
            "role": "system",
            "content": f"Neural processing error: {str(e)}"
        })

def start_server():
    """Start the server with proper error handling"""
    try:
        print(f"\n🌟 Starting FractiCognition Web Interface...")
        print(f"📱 Opening browser to: http://localhost:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        alt_port = find_free_port()
        print(f"\n🔄 Retrying on port {alt_port}...")
        try:
            global HTML_CONTENT
            HTML_CONTENT = HTML_CONTENT.replace(
                f'ws://localhost:{port}/ws',
                f'ws://localhost:{alt_port}/ws'
            )
            print(f"📱 Opening browser to: http://localhost:{alt_port}")
            uvicorn.run(app, host="0.0.0.0", port=alt_port, log_level="info")
        except Exception as e:
            print(f"\n❌ Failed to start server: {e}")

if __name__ == "__main__":
    start_server() 