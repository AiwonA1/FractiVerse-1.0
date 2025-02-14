"""
🚀 FractiCody 1.0 - Main Entry Point
Handles Fractal AI Initialization and System Coordination
"""
import os
from fracticody_engine import FractiCodyEngine
from config import settings
from fastapi import FastAPI
import uvicorn

# 🔹 Initialize FastAPI App
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "✅ FractiCody 1.0 is running!"}

if __name__ == "__main__":
    print("🔹 Initializing FractiCody 1.0...")

    fracti_ai = FractiCodyEngine()
    fracti_ai.start()
    
    print("✅ FractiCody 1.0 is now running.")

    # 🔹 Set the port dynamically based on Render's environment
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if not set
    uvicorn.run(app, host="0.0.0.0", port=port)
