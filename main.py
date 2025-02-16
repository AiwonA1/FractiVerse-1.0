import sys
import os
from fracticody_engine import FractiCodyEngine
from config import settings
from fastapi import FastAPI
import uvicorn

# Ensure Python recognizes 'core/' as a package
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "core")))

from core.fractal_cognition import FractalCognition
from core.memory_manager import MemoryManager
from core.fracti_fpu import FractiProcessingUnit

# Initialize FastAPI App
app = FastAPI()

# Persistent Global Instance of FractiCody
fracticody = FractiCodyEngine()

@app.post("/command")
def command(request: dict):
    user_input = request.get("command", "").strip()
    if not user_input:
        return {"error": "Invalid input. Command is required."}
    
    response = fracticody.process_input(user_input)
    return {"response": response}

if __name__ == "__main__":
    print("🔹 Initializing FractiCody 1.0...")
    
    fracticody.start()  # Ensure the engine starts correctly
    
    print("✅ FractiCody 1.0 is now running.")

    # Set the port dynamically based on the environment
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if not set
    uvicorn.run(app, host="0.0.0.0", port=port)
    fracti_ai.start()
    
    print("✅ FractiCody 1.0 is now running.")

    # Set the port dynamically based on Render's environment
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if not set
    uvicorn.run(app, host="0.0.0.0", port=port)
