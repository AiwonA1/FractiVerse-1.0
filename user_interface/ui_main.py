"""
🖥️ FractiGator UI - Main User Interface for FractiCody
Handles UI logic, user interactions, and real-time dashboard updates.
"""

import os
import uvicorn
from fastapi import FastAPI
from fractigator_navigation import RealityNavigator
from active_projects import ActiveProjects
from daily_suggestions import DailySuggestions

# Initialize FastAPI app
app = FastAPI()

class FractiGatorUI:
    def __init__(self):
        self.navigator = RealityNavigator()
        self.projects = ActiveProjects()
        self.suggestions = DailySuggestions()
        self.fracti_touch_enabled = self.check_fracti_touch_status()

    def check_fracti_touch_status(self):
        """Checks if FractiTouch is enabled in user settings."""
        return bool(os.environ.get("FRACTITOUCH_ENABLED", "False").lower() == "true")

    def detect_device(self):
        """Detects device type to optimize UI rendering."""
        user_agent = os.environ.get("DEVICE_USER_AGENT", "Unknown")
        if "iPhone" in user_agent or "Android" in user_agent:
            return "📱 Mobile Optimized UI"
        elif "Mac" in user_agent or "Windows" in user_agent:
            return "💻 Desktop Optimized UI"
        else:
            return "🔍 Generic UI Layout"

    def launch(self):
        """Starts the UI and displays the user dashboard."""
        print("🔹 Welcome to FractiCody 1.0 UI")
        print("✨ Visit | Explore | Discover | Create")
        print(self.detect_device())
        print(f"🖐 FractiTouch Enabled: {self.fracti_touch_enabled}")
        print(self.navigator.get_current_reality())
        print(self.projects.list_active_projects())
        print(self.suggestions.get_daily_suggestion())

# Define API Endpoints
@app.get("/")
def home():
    return {
        "message": "✅ FractiCody 1.0 UI is running!",
        "ui_options": ["Visit", "Explore", "Discover", "Create"],
        "device_optimization": FractiGatorUI().detect_device(),
        "fracti_touch_status": FractiGatorUI().fracti_touch_enabled,
        "current_reality": RealityNavigator().get_current_reality(),
        "active_projects": ActiveProjects().list_active_projects(),
        "daily_suggestion": DailySuggestions().get_daily_suggestion(),
    }

if __name__ == "__main__":
    ui = FractiGatorUI()
    ui.launch()

    # Set the port dynamically based on Render's environment (default: 8080)
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
