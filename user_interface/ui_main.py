"""
🖥️ FractiGator 1.0 - Main User Interface for FractiCody
Handles UI logic, user interactions, and real-time dashboard updates.
"""

import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fractigator_navigation import RealityNavigator
from active_projects import ActiveProjects
from daily_suggestions import DailySuggestions
from ui_navigation import UI_Navigation  # New UI switching module

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="user_interface/templates")

class FractiGatorUI:
    def __init__(self):
        self.navigator = RealityNavigator()
        self.projects = ActiveProjects()
        self.suggestions = DailySuggestions()
        self.ui_navigation = UI_Navigation()  # UI switching module

    def launch(self):
        """Starts the UI and displays the user dashboard."""
        print("🔹 Welcome to FractiCody 1.0 UI")
        print("🔹 [ FractiGator 1.0 ▼ ] - Click to expand menu")
        print(self.navigator.get_current_reality())
        print(self.projects.list_active_projects())
        print(self.suggestions.get_daily_suggestion())

# Define API Endpoints
@app.get("/")
def home(request: Request):
    """Serves the main UI dashboard with FractiCody status."""
    data = {
        "request": request,
        "message": "✅ FractiCody 1.0 UI is running!",
        "current_reality": RealityNavigator().get_current_reality(),
        "active_projects": ActiveProjects().list_active_projects(),
        "daily_suggestion": DailySuggestions().get_daily_suggestion(),
        "available_displays": UI_Navigation().get_available_displays()
    }
    return templates.TemplateResponse("index.html", data)

@app.get("/switch/{display}")
def switch_display(display: str):
    """Allows switching between UI displays."""
    return UI_Navigation().switch_display(display)

if __name__ == "__main__":
    ui = FractiGatorUI()
    ui.launch()

    # Set the port dynamically based on Render's environment
    port = int(os.environ.get("PORT", 8080))  # Default to 8080
    uvicorn.run(app, host="0.0.0.0", port=port)
