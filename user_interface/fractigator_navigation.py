"""
🌍 FractiGator Navigation - Reality Navigation & UI Adaptation
Handles seamless transitions between FractiVerse, LinearVerse, and Alternate Realities.
"""

class RealityNavigator:
    def __init__(self):
        self.realities = {
            "FractiVerse": "🌌 Infinite Fractal Knowledge Expansion",
            "LinearVerse": "📊 Structured AI Processing Mode",
            "Alternate Realities": "🔀 AI-Driven Experimental Simulations",
        }
        self.current_reality = "LinearVerse"  # Default starting reality

    def switch_reality(self, new_reality):
        """Switches FractiCody's interface to the specified reality mode."""
        if new_reality in self.realities:
            self.current_reality = new_reality
            return f"🔄 Switched to {new_reality}: {self.realities[new_reality]}"
        return "❌ Invalid Reality Selection"

    def get_current_reality(self):
        """Returns the currently active reality mode."""
        return f"🌀 Current Reality: {self.current_reality}"

    def get_available_realities(self):
        """Lists all available realities for user navigation."""
        return self.realities

if __name__ == "__main__":
    navigator = RealityNavigator()
    print(navigator.get_current_reality())
    print(navigator.switch_reality("FractiVerse"))
    print(navigator.get_available_realities())
