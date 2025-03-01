import asyncio
import websockets
import json
from core.fracticognition import FractiCognition
import matplotlib.pyplot as plt
from tqdm import tqdm

async def test_fpu_expansion():
    """Test expansion to 1000 FPU through interactive UI"""
    
    uri = "ws://localhost:8003/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("🚀 Connected to FractiCognition system")
            
            # Send expansion command
            command = {
                "type": "command",
                "content": "expand cognitive capacity to 1000 FPU"
            }
            await websocket.send(json.dumps(command))
            
            # Track progress through responses
            while True:
                response = await websocket.recv()
                response_data = json.loads(response)
                
                if "error" in response_data:
                    print(f"❌ Error: {response_data['error']}")
                    break
                    
                print(response_data["content"])
                
                # Check for completion
                if "Development complete" in response_data["content"]:
                    print("🎉 Reached 1000 FPU!")
                    break
                    
                # Check for stage advancement
                if "Advanced to" in response_data["content"]:
                    print(f"🎓 {response_data['content']}")
                
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_fpu_expansion()) 