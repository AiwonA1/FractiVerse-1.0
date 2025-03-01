import asyncio
import os
import sys
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
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
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
        return port

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>FractiCognition 1.0</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
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
        #chat-window {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background: #1e1e1e;
            color: #fff;
            border-radius: 5px;
            margin-bottom: 20px;
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
        .commands {
            background: #fff3e0;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 FractiCognition 1.0</h1>
        
        <div class="status-bar">
            <span>Neural Capacity: <span id="capacity">0.01%</span></span>
            <span>Active Patterns: <span id="patterns">0</span></span>
            <span>Processing State: <span id="state">Ready</span></span>
        </div>

        <div class="commands">
            <h3>Available Commands:</h3>
            <ul>
                <li><code>/expand</code> - Grow neural capacity</li>
                <li><code>/status</code> - Show neural network status</li>
                <li><code>/clear</code> - Clear conversation</li>
            </ul>
        </div>

        <div id="neural-status">
            Initializing neural network...
        </div>

        <div id="neural-viz">
            <canvas id="activity-viz" width="960" height="130"></canvas>
        </div>

        <div id="chat-window"></div>

        <input type="text" id="input" placeholder="Chat with FractiCognition or use commands (e.g. /expand, /status)">
    </div>

    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        const chatWindow = document.getElementById("chat-window");
        const neuralStatus = document.getElementById("neural-status");
        const canvas = document.getElementById("activity-viz");
        const ctx = canvas.getContext("2d");
        
        // Neural activity visualization
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

        function addMessage(text, className) {
            const msg = document.createElement("div");
            msg.textContent = text;
            msg.className = `message ${className}`;
            chatWindow.appendChild(msg);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
        
        ws.onopen = () => {
            neuralStatus.innerHTML = "✅ Connected to FractiCognition Neural Network";
            addMessage("Neural network connection established", "system");
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if(data.type === 'neural_activity') {
                updateViz(data.activity_level);
                document.getElementById("capacity").textContent = (data.fpu_level * 100).toFixed(2) + '%';
                document.getElementById("patterns").textContent = data.active_patterns;
                document.getElementById("state").textContent = data.processing_state;
                
                addMessage(`🔄 ${data.description}`, "neural-activity");
            }
            else if(data.type === 'message') {
                addMessage(data.content, data.role);
            }
            else if(data.type === 'status') {
                neuralStatus.innerHTML = data.content;
            }
        };
        
        document.getElementById("input").onkeypress = (e) => {
            if(e.key === 'Enter') {
                const input = e.target;
                const message = input.value.trim();
                
                if(message) {
                    addMessage(`You: ${message}`, "user");
                    ws.send(message);
                    input.value = "";
                }
            }
        };
    </script>
</body>
</html>
"""

def create_web_interface(cognitive_system):
    """Create and launch the web interface"""
    app = FastAPI()
    port = find_free_port()
    
    # Register routes
    @app.get("/")
    async def get():
        return HTMLResponse(HTML_CONTENT)
        
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        
        try:
            while True:
                message = await websocket.receive_text()
                
                if message.startswith('/'):
                    # Handle commands
                    await handle_command(websocket, message[1:], cognitive_system)
                else:
                    # Natural language interaction
                    response = await cognitive_system.process_natural_language(message)
                    await websocket.send_text(response)
                    
        except WebSocketDisconnect:
            print("Client disconnected")
    
    # Launch browser after delay
    def launch_browser():
        time.sleep(2)
        url = f"http://localhost:{port}"
        print(f"\n🌐 Opening neural interface at: {url}")
        webbrowser.open(url)
    
    # Start server
    try:
        print(f"\n🚀 Starting neural interface on port {port}...")
        
        # Launch browser in background
        import threading
        threading.Thread(target=launch_browser, daemon=True).start()
        
        # Start server with proper config
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            reload=False
        )
        server = uvicorn.Server(config)
        server.run()
        
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        raise

async def handle_command(websocket: WebSocket, command: str, cognitive_system: FractalCognition):
    try:
        if command == 'expand':
            print("\n🧠 Starting neural expansion...")
            print(f"Current FPU Level: {cognitive_system.fpu_level * 100:.1f}%")
            
            async for status in cognitive_system.expand_cognitive_capacity(100):
                metrics = status.get('metrics', {})
                progress = f"""
                Neural Growth Progress:
                - Current FPU Level: {status['fpu_level']*100:.1f}%
                - Target FPU Level: 100%
                - Processing Speed: {metrics.get('processing', 0):.1f}%
                - Pattern Analysis: {metrics.get('pattern_depth', 0):.1f}%
                - Learning Rate: {metrics.get('learning', 0):.1f}%
                - Memory Integration: {metrics.get('memory', 0):.1f}%
                - Active Patterns: {status['patterns']}
                """
                print(f"📈 {progress}")
                
                await websocket.send_json({
                    "type": "neural_activity",
                    "activity_level": status['fpu_level'],
                    "fpu_level": status['fpu_level'],
                    "target_fpu": 1.0,  # 100%
                    "active_patterns": status['patterns'],
                    "processing_state": "Expanding",
                    "description": progress
                })
                
            print("✨ Neural expansion complete!")
            print(f"Final FPU Level: {cognitive_system.fpu_level * 100:.1f}%")
            
            # Send final status update
            await websocket.send_json({
                "type": "status",
                "content": f"""
                <h3>Neural Expansion Complete</h3>
                <p>Current FPU Level: {cognitive_system.fpu_level * 100:.1f}%</p>
                <p>Active Patterns: {len(cognitive_system.pattern_network.patterns)}</p>
                <p>Processing Speed: {cognitive_system.fpu_metrics['processing_speed'] * 100:.1f}%</p>
                <p>Pattern Analysis: {cognitive_system.fpu_metrics['pattern_depth'] * 100:.1f}%</p>
                <p>Learning Rate: {cognitive_system.fpu_metrics['learning_efficiency'] * 100:.1f}%</p>
                <p>Memory Integration: {cognitive_system.fpu_metrics['memory_integration'] * 100:.1f}%</p>
                """
            })
            
        elif command == 'status':
            status = cognitive_system.get_status()
            status_text = f"""
            <h3>Neural Network Status:</h3>
            <p>📊 FPU Level: {status['fpu_level']*100:.1f}%</p>
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

async def process_neural_input(websocket: WebSocket, message: str, cognitive_system: FractalCognition):
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

if __name__ == "__main__":
    try:
        print(f"\n🌟 Starting FractiCognition Web Interface...")
        print(f"📱 Opening browser to: http://localhost:{port}")
        
        # Launch browser in separate thread
        import threading
        threading.Thread(target=launch_browser, args=(port,), daemon=True).start()
        
        # Start server
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        print(f"\n❌ Server error: {e}") 