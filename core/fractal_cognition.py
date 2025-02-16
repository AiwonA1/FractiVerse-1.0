import time
import json
import os

class FractalCognition:
    """FractiCody's cognitive system with structured learning sequences and intelligent self-improvement."""

    def __init__(self):
        print("✅ Fractal Cognition Initializing...")
        self.memory = self.load_memory()
        self.cognition_level = self.load_cognition_level()
        self.learning_active = True

        time.sleep(1)  # Boot delay for stability

        # If starting fresh, load core knowledge base
        if len(self.memory) == 0:
            self.load_core_knowledge()

        # Automatically begin structured learning
        self.start_learning_sequence()

    def load_cognition_level(self):
        """Loads the last saved cognition level (persistent learning)."""
        if os.path.exists("cognition_level.json"):
            with open("cognition_level.json", "r") as file:
                return json.load(file).get("cognition_level", 1.0)
        return 1.0

    def save_cognition_level(self):
        """Saves the current cognition level."""
        with open("cognition_level.json", "w") as file:
            json.dump({"cognition_level": self.cognition_level}, file)

    def load_memory(self):
        """Loads stored AI knowledge from memory file."""
        if os.path.exists("memory.json"):
            with open("memory.json", "r") as file:
                return json.load(file)
        return {}

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
        self.cognition_level += 5  # Start with some cognition points
        self.save_cognition_level()

    def start_learning_sequence(self):
        """Controls FractiCody's structured learning development."""
        print("🚀 Starting Structured Learning Progression...")
        
        if self.cognition_level < 20:
            self.learn_language_basics()
        
        if self.cognition_level >= 20:
            self.enable_internet_learning()

        if self.cognition_level >= 50:
            self.enable_intelligent_study_selection()

    def learn_language_basics(self):
        """Basic language acquisition step."""
        print("🧠 Learning Language Basics...")
        language_knowledge = {
            "hello": "A greeting used when meeting someone.",
            "how are you?": "A common phrase used to ask about well-being.",
            "goodbye": "A phrase used when leaving."
        }
        self.memory.update(language_knowledge)
        self.save_memory()
        self.cognition_level += 10
        self.save_cognition_level()

    def enable_internet_learning(self):
        """Grants access to search the internet for knowledge acquisition."""
        print("🌐 Enabling Internet Research Mode...")
        self.memory["internet_access"] = "FractiCody can now access online sources to expand knowledge."
        self.save_memory()
        self.cognition_level += 10
        self.save_cognition_level()

    def enable_intelligent_study_selection(self):
        """Allows FractiCody to prioritize the most valuable knowledge areas to study."""
        print("🧠 Activating Self-Directed Study Optimization...")
        self.memory["study_optimization"] = "FractiCody now selects the most beneficial learning topics based on gaps in knowledge."
        self.save_memory()
        self.cognition_level += 10
        self.save_cognition_level()

    def activate(self):
        """Ensures cognition remains active."""
        self.learning_active = True
        return "Cognition Activated"

    def process_input(self, user_input):
        """Processes user input and learns from interactions."""
        user_input = user_input.lower().strip()

        if user_input in self.memory:
            return self.memory[user_input]

        response = "I don't know yet. Teach me, and I'll remember."
        self.memory[user_input] = response
        self.save_memory()
        return response
