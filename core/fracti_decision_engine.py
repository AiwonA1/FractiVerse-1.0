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
        if os.path.exists("decision_memory.json"):
            try:
                with open("decision_memory.json", "r") as file:
                    return json.load(file)
            except json.JSONDecodeError:
                return {}  # Return empty if the file is corrupted
        return {}

    def save_decision_memory(self):
        """Saves decision-making data to memory."""
        with open("decision_memory.json", "w") as file:
            json.dump(self.decisions, file)

    def process_decision(self, context, options):
        """Processes decision-making based on learned logic."""
        context = context.lower().strip()
        
        if context in self.decisions:
            return self.decisions[context]  # Return stored decision

        # If no known decision, ask for learning input
        response = f"I don't have a decision for '{context}' yet. What should I do?"
        self.decisions[context] = response
        self.save_decision_memory()
        
        return response
