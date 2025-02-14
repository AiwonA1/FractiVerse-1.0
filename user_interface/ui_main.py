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

    def launch(self):
        """Starts the UI and displays the user dashboard."""
        print("🔹 Welcome to FractiCody 1.0 UI")
        print(self.navigator.get_current_reality())
        print(self.projects.list_active_projects())
        print(self.suggestions.get_daily_suggestion())

# Define API Endpoints
@app.get("/")
def home():
    return {
        "message": "✅ FractiCody 1.0 UI is running!",
        "current_reality": RealityNavigator().get_current_reality(),
        "active_projects": ActiveProjects().list_active_projects(),
        "daily_suggestion": DailySuggestions().get_daily_suggestion(),
    }

if __name__ == "__main__":
    ui = FractiGatorUI()
    ui.launch()

    # Set the port dynamically based on Render's environment
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if not set
    uvicorn.run(app, host="0.0.0.0", port=port)

