import asyncio
import os
import sys
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import socket
from contextlib import closing

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

# Initialize FastAPI and cognitive system
app = FastAPI()
port = find_free_port()  # Get an available port
cognitive_system = FractalCognition()

# Initialize the system
print("\n🧠 Initializing FractiCognition...")
cognitive_system.bootstrap_cognition()
cognitive_system.initialize_core_knowledge()

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>FractiCognition Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #messages { margin: 20px 0; padding: 10px; border: 1px solid #ccc; height: 300px; overflow-y: auto; }
        #input { width: 100%; padding: 5px; }
        .message { margin: 5px 0; }
        .user { color: blue; }
        .bot { color: green; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>FractiCognition Simple Test</h1>
    <div id="status">Connecting...</div>
    <div id="messages"></div>
    <input type="text" id="input" placeholder="Type a message and press Enter">

    <script>
        const messages = document.getElementById('messages');
        const status = document.getElementById('status');
        const input = document.getElementById('input');
        
        function addMessage(text, className) {
            const msg = document.createElement('div');
            msg.textContent = text;
            msg.className = `message ${className}`;
            messages.appendChild(msg);
            messages.scrollTop = messages.scrollHeight;
            console.log(`${className}:`, text);
        }
        
        // Use current host for WebSocket
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = () => {
            status.textContent = '✅ Connected to FractiCognition';
            status.style.color = 'green';
            addMessage('Connected to neural network', 'system');
        };
        
        ws.onmessage = (event) => {
            addMessage(event.data, 'bot');
        };
        
        ws.onclose = () => {
            status.textContent = '❌ Disconnected';
            status.style.color = 'red';
            addMessage('Disconnected from neural network', 'system');
        };
        
        ws.onerror = (error) => {
            status.textContent = '❌ Connection Error';
            status.style.color = 'red';
            addMessage('Connection error: ' + error.message, 'error');
        };
        
        input.onkeypress = (e) => {
            if (e.key === 'Enter' && input.value.trim()) {
                const message = input.value;
                addMessage('You: ' + message, 'user');
                ws.send(message);
                input.value = '';
            }
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
    print("\n✅ WebSocket connected")
    
    try:
        while True:
            # Receive message
            message = await websocket.receive_text()
            print(f"\n👤 Received: {message}")
            
            try:
                # Process through neural network
                print("🤔 Processing through neural network...")
                response = await cognitive_system.process_command(message)
                print(f"🧠 Response: {response}")
                
                # Send response
                await websocket.send_text(f"FractiCognition: {response}")
                
            except Exception as e:
                error_msg = f"Neural processing error: {str(e)}"
                print(f"❌ {error_msg}")
                await websocket.send_text(f"Error: {error_msg}")
            
    except Exception as e:
        print(f"\n❌ WebSocket error: {e}")
    finally:
        print("\n👋 WebSocket disconnected")

if __name__ == "__main__":
    try:
        print(f"\n🌟 Starting simple test server on port {port}...")
        print(f"📱 Open your browser to: http://localhost:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        # Try alternate port if first one fails
        alt_port = find_free_port()
        print(f"\n🔄 Retrying on port {alt_port}...")
        try:
            print(f"📱 Open your browser to: http://localhost:{alt_port}")
            uvicorn.run(app, host="0.0.0.0", port=alt_port)
        except Exception as e:
            print(f"\n❌ Failed to start server: {e}") 