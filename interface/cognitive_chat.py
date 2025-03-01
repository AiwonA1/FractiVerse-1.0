from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio
import json
import webbrowser
import uvicorn
import socket
from datetime import datetime
import time
from core.fractal_cognition import FractalCognition

app = FastAPI()

# Initialize FractiCognition
cognitive_system = FractalCognition()

# Web interface HTML with ChatGPT-like styling
HTML_CONTENT = """
<!DOCTYPE html>
<html>
    <head>
        <title>FractiCognition 1.0</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                display: flex;
                height: 100vh;
                background: #343541;
                color: #ECECF1;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }

            .sidebar {
                width: 260px;
                background: #202123;
                padding: 10px;
                display: flex;
                flex-direction: column;
            }

            .main-content {
                flex: 1;
                display: flex;
                flex-direction: column;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }

            #chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px 0;
            }

            .message {
                display: flex;
                padding: 20px;
                gap: 20px;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }

            .user-message {
                background: #343541;
            }

            .cognitive-message {
                background: #444654;
            }

            .message-content {
                flex: 1;
                line-height: 1.5;
            }

            .input-container {
                position: fixed;
                bottom: 0;
                left: 260px;
                right: 0;
                padding: 20px;
                background: #343541;
                display: flex;
                align-items: center;
                max-width: 1000px;
                margin: 0 auto;
            }

            #messageInput {
                flex: 1;
                padding: 12px;
                background: #40414F;
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 4px;
                color: white;
                font-size: 16px;
            }

            .send-button {
                background: #19C37D;
                border: none;
                border-radius: 4px;
                padding: 12px 20px;
                margin-left: 10px;
                color: white;
                cursor: pointer;
            }

            .metrics {
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(0,0,0,0.7);
                padding: 10px;
                border-radius: 4px;
                font-size: 12px;
            }

            .new-chat-btn {
                border: 1px solid #4E4F60;
                border-radius: 4px;
                padding: 12px;
                color: #ECECF1;
                background: transparent;
                cursor: pointer;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 12px;
            }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <button class="new-chat-btn" onclick="newChat()">
                <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" height="24" width="24"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                New Chat
            </button>
        </div>

        <div class="main-content">
            <div id="chat-messages"></div>
            <div class="input-container">
                <input type="text" id="messageInput" placeholder="Send a message..." onkeypress="handleKeyPress(event)">
                <button class="send-button" onclick="sendMessage()">Send</button>
            </div>
            <div id="metrics" class="metrics"></div>
        </div>

        <script>
            let ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = function() {
                console.log("Connected to FractiCognition");
            };
            
            ws.onmessage = function(event) {
                const response = JSON.parse(event.data);
                if (response.type === 'cognitive_response') {
                    addMessage(response.content, 'cognitive');
                    updateMetrics(response.metrics);
                }
            };
            
            ws.onerror = function(error) {
                console.error("WebSocket error:", error);
                addMessage("Connection error. Please try again.", 'system');
            };

            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (message) {
                    addMessage(message, 'user');
                    ws.send(message);
                    input.value = '';
                }
            }

            function handleKeyPress(event) {
                if (event.keyCode === 13 && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                }
            }
            
            function addMessage(content, type) {
                const messages = document.getElementById('chat-messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}-message`;
                
                const avatar = document.createElement('div');
                avatar.className = 'avatar';
                avatar.innerHTML = type === 'user' ? '👤' : '🤖';
                
                const messageContent = document.createElement('div');
                messageContent.className = 'message-content';
                messageContent.textContent = content;
                
                messageDiv.appendChild(avatar);
                messageDiv.appendChild(messageContent);
                messages.appendChild(messageDiv);
                
                messages.scrollTop = messages.scrollHeight;
            }
            
            function updateMetrics(metrics) {
                const metricsDiv = document.getElementById('metrics');
                metricsDiv.innerHTML = `
                    Coherence: ${(metrics.coherence_strength * 100).toFixed(1)}%<br>
                    Memory: ${(metrics.quantum_memory * 100).toFixed(1)}%<br>
                    Resonance: ${(metrics.pattern_resonance * 100).toFixed(1)}%
                `;
            }

            function newChat() {
                document.getElementById('chat-messages').innerHTML = '';
                document.getElementById('metrics').innerHTML = '';
            }

            // Focus input on load
            document.getElementById('messageInput').focus();
        </script>
    </body>
</html>
"""

def find_free_port():
    """Find an available port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def launch_browser(port: int):
    """Launch browser after short delay"""
    import time
    time.sleep(1.5)  # Wait for server to start
    webbrowser.open(f'http://localhost:{port}')

async def start_interface():
    """Start the cognitive interface"""
    try:
        port = find_free_port()
        
        print(f"\n🌟 Starting FractiCognition Interface on port {port}")
        print("✨ Initializing quantum neural fabric...")
        
        # Launch browser in background
        import threading
        threading.Thread(target=lambda: launch_browser(port), daemon=True).start()
        
        # Start server
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        print(f"\n❌ Interface error: {str(e)}")
        raise

@app.get("/")
async def get():
    return HTMLResponse(HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("\n✅ Cognitive connection established")
    
    try:
        while True:
            # Receive message
            message = await websocket.receive_text()
            print(f"\n👤 User: {message}")
            
            try:
                # Process through cognitive system
                response = await cognitive_system.process_natural_language(message)
                
                # Send response with metrics
                await websocket.send_json({
                    "type": "cognitive_response",
                    "content": response,
                    "metrics": {
                        "coherence_strength": cognitive_system.quantum_metrics['coherence_strength'],
                        "quantum_memory": cognitive_system.quantum_metrics['quantum_memory'],
                        "pattern_resonance": cognitive_system.quantum_metrics['pattern_resonance']
                    }
                })
                
            except Exception as e:
                print(f"Processing error: {e}")
                await websocket.send_json({
                    "type": "cognitive_response", 
                    "content": f"I encountered an error: {str(e)}",
                    "metrics": {
                        "coherence_strength": 0.01,
                        "quantum_memory": 0.01,
                        "pattern_resonance": 0.01
                    }
                })
                
    except Exception as e:
        print(f"\n❌ WebSocket error: {str(e)}")

# Auto-start when run directly
if __name__ == "__main__":
    asyncio.run(start_interface()) 