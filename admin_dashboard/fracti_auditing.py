"""
🛠️ FractiAdmin 1.0 - Admin Control Panel for FractiCody
Allows full ecosystem control, AI governance, and system monitoring.
"""

import os
import uvicorn
from fastapi import FastAPI
from ui_navigation import UI_Navigation  # UI switching module

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "✅ FractiAdmin 1.0 is running!",
        "admin_controls": [
            "Monitor AI Nodes",
            "Manage FractiChain Transactions",
            "Issue FractiTokens",
            "Adjust System Parameters",
            "Broadcast Messages"
        ],
        "available_displays": UI_Navigation().get_available_displays()
    }

@app.get("/switch/{display}")
def switch_display(display: str):
    return UI_Navigation().switch_display(display)

if __name__ == "__main__":
    print("🔹 [ FractiAdmin 1.0 ▼ ] - Click to expand menu")

    # Set the port dynamically based on Render's environment
    port = int(os.environ.get("PORT", 8181))  # Default to 8181
    uvicorn.run(app, host="0.0.0.0", port=port)
