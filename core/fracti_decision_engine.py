import time
import json
import os

class FractiDecisionEngine:
    """FractiCody's decision-making system, handling logic, reasoning, and adaptive decisions."""

    def __init__(self):
        print("✅ FractiDecisionEngine Initialized...")
        self.decisions = self.load_decision_memory()
        self.decision_level = self.load_decision_level()  # Persistent decision tracking
        self.adaptive_learning = True

        # Ensure smooth initialization timing
        time.sleep(1)

    def load_decision_level(self):
        """Loads the last saved decision level (persistent decision-making memory)."""
        if os.path.exists("decision_level.json"):
            with open("decision_level.json", "r") as file:
                return json.load(file).get("decision_level", 1.0)
        return 1.0  # Default starting level

    def save_decision_level(self):
        """Saves the current decision level to prevent resets."""
        with open("decision_level.json", "w") as file:
            json.dump({"decision_level": self.decision_level}, file)

    def load_decision_memory(self):
        """Loads stored decision-making data."""
        if os.path.exists("dec
