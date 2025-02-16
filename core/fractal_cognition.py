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
        """Processes user input and learns from interactions."""
        user_input = user_input.lower().strip()

        if user_input in self.memory:
            return self.memory[user_input]

        response = "I don't know yet. Teach me, and I'll remember."
        self.memory[user_input] = response
        self.cognition_level += 0.10
        self.save_memory()
        self.save_cognition_level()
        return response
                return json.load(file).get("cognition_level", 1.0)
        return 1.0  # Default cognition level

    def save_cognition_level(self):
        """Saves the current cognition level."""
        with open("cognition_level.json", "w") as file:
            json.dump({"cognition_level": self.cognition_level}, file)

    def load_memory(self):
        """Loads stored AI knowledge from memory file."""
        if os.path.exists("memory.json"):
            with open("memory.json", "r") as file:
                return json.load(file)
        return {}  # Start with an empty memory dictionary if no prior learning

    def save_memory(self):
        """Saves AI knowledge persistently."""
        with open("memory.json", "w") as file:
            json.dump(self.memory, file)

    def load_core_knowledge(self):
        """Preloads fundamental knowledge on first startup."""
        print("📚 Loading Core Knowledge Base...")
        core_knowledge = {
            "hello": "Hello! I am FractiCody, an evolving AI.",
            "who are you?": "I am FractiCody, a self-learning AI designed to evolve over time.",
            "what is AI?": "AI stands for Artificial Intelligence, the simulation of human intelligence in machines.",
            "what is the sun?": "The sun is a star at the center of the Solar System.",
            "how do you learn?": "I learn through user interactions, memory storage, and deep learning acceleration."
        }
        self.memory.update(core_knowledge)
        self.save_memory()
        self.cognition_level += 5  # Start with a baseline cognition level
        self.save_cognition_level()

    def activate(self):
        """Ensures cognition remains active."""
        self.learning_active = True
        return "Cognition Activated"

    def process_input(self, user_input):
        """Processes user input and learns from interactions."""
        user_input = user_input.lower().strip()

        # Retrieve known responses from memory
        if user_input in self.memory:
            return self.memory[user_input]

        # If new input, store it and prepare to learn
        response = "I don't know yet. Teach me, and I'll remember."
        self.memory[user_input] = response
        self.save_memory()
        return response
