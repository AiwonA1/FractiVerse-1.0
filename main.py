""import sys
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
from core.fracti_fpu import FractiFPU

# Initialize FastAPI App
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "✅ FractiCody 1.0 is running!"}

if __name__ == "__main__":
    print("🔹 Initializing FractiCody 1.0...")
    
    fracti_ai = FractiCodyEngine()
    fracti_ai.start()
    
    print("✅ FractiCody 1.0 is now running.")

    # Set the port dynamically based on Render's environment
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if not set
    uvicorn.run(app, host="0.0.0.0", port=port)
