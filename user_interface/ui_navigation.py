"""
📌 UI Navigation Module
Handles UI switching between FractiGator & FractiAdmin.
"""

class UI_Navigation:
    def __init__(self):
        self.available_views = ["FractiGator", "FractiAdmin"]
        self.current_view = "FractiGator"

    def switch_view(self, view):
        """Switches the UI view."""
        if view in self.available_views:
            self.current_view = view
            return f"🔄 Switched to {view} UI"
        return "❌ Invalid UI Option"

    def get_current_view(self):
        """Returns the current active UI view."""
        return self.current_view

    def get_available_displays(self):
        """Returns a list of all available UI displays."""
        return self.available_views  # This method was missing and is now added
