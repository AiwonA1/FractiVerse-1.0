"""
🧭 FractiGator Navigation - Reality Toggling System
Allows users to switch between FractiVerse, LinearVerse, and Alternate Realities.
"""
class RealityNavigator:
    def __init__(self):
        self.current_reality = "LinearVerse"

    def switch_reality(self, new_reality):
        """Switches the active AI reality."""
        realities = ["FractiVerse", "LinearVerse", "Alternate Reality"]
        if new_reality in realities:
            self.current_reality = new_reality
            return f"🌌 Reality Switched to: {new_reality}"
        return "❌ Invalid Reality Selection"

    def get_current_reality(self):
        """Returns the currently active AI reality."""
        return f"🌀 Current Reality: {self.current_reality}"

if __name__ == "__main__":
    navigator = RealityNavigator()
    print(navigator.switch_reality("FractiVerse"))
    print(navigator.get_current_reality())
