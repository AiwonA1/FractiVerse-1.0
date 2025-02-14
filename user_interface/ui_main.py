"""
🖥️ FractiGator UI - Main User Interface for FractiCody
Handles UI logic, user interactions, and real-time dashboard updates.
"""
from fractigator_navigation import RealityNavigator
from active_projects import ActiveProjects
from daily_suggestions import DailySuggestions

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

if __name__ == "__main__":
    ui = FractiGatorUI()
    ui.launch()
