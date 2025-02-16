import time
import json
import os

class FractalCognition:
    """FractiCody's cognitive system with structured learning sequences and persistent memory."""

    def __init__(self):
        print("✅ Fractal Cognition Initializing...")
        self.memory = self.load_memory()
        self.cognition_level = self.load_cognition_level()
        self.learning_active = True

        time.sleep(1)  # Boot delay for stability

        if not self.memory:
            self.load_core_knowledge()

    def load_cognition_level(self):
        """Loads the last saved cognition level."""
        if os.path.exists("cognition_level.json"):
            try:
                with open("cognition_level.json", "r") as file:
                    return json.load(file).get("cognition_level", 1.0)
            except json.JSONDecodeError:
                return 1.0
        return 1.0

    def save_cognition_level(self):
        """Saves the cognition level."""
        with open("cognition_level.json", "w") as file:
            json.dump({"cognition_level": self.cognition_level}, file, indent=4)

    def load_memory(self):
        """Loads stored AI knowledge."""
        if os.path.exists("memory.json"):
            try:
                with open("memory.json", "r") as file:
                    return json.load(file)
            except json.JSONDecodeError:
                return {}
        return {}

    def save_memory(self):
        """Saves AI knowledge."""
        with open("memory.json", "w") as file:
            json.dump(self.memory, file, indent=4)

    def process_input(self, user_input):
        """Processes user input and prioritizes retrieving human-like responses."""
        user_input = user_input.lower().strip()

        # ✅ Step 1: Return the answer if already learned
        if user_input in self.memory:
            return self.memory[user_input]  # ✅ Now only returns real responses

        # ✅ Step 2: If unknown, provide a more conversational response
        response = "I haven't learned that yet. Can you teach me?"
        self.memory[user_input] = response
        self.cognition_level += 0.10  # Cognition updates in background
        self.save_memory()
        self.save_cognition_level()
    
        return response  # ✅ Now returns conversational response, not cognition updates
