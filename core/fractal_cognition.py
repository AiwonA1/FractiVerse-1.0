import time
import json
import os

class FractalCognition:
    """FractiCody's core cognition system with persistent memory and structured learning."""

    def __init__(self):
        print("✅ Fractal Cognition Initializing...")
        self.memory = self.load_memory()
        self.cognition_level = self.load_cognition_level()  # Persistent cognition
        self.learning_active = True
        self.current_stage = self.determine_learning_stage()

        # Ensure safe boot timing
        time.sleep(1)

        # Automatically boot deep learning
        self.start_self_learning()

    def load_cognition_level(self):
        """Loads the last saved cognition level from memory (persistent learning)."""
        if os.path.exists("cognition_level.json"):
            with open("cognition_level.json", "r") as file:
                return json.load(file).get("cognition_level", 1.0)
        return 1.0  # Default starting cognition level

    def save_cognition_level(self):
        """Saves the current cognition level to prevent resets on restart."""
        with open("cognition_level.json", "w") as file:
            json.dump({"cognition_level": self.cognition_level}, file)

    def load_memory(self):
        """Loads stored AI knowledge from memory file."""
        if os.path.exists("memory.json"):
            with open("memory.json", "r") as file:
                return json.load(file)
        return {}  # Empty memory if no prior learning

    def save_memory(self):
        """Saves AI knowledge persistently."""
        with open("memory.json", "w") as file:
            json.dump(self.memory, file)

    def determine_learning_stage(self):
        """Determines FractiCody's learning stage based on cognition level."""
        if self.cognition_level < 5:
            return "Infant Learning (Basic Words)"
        elif self.cognition_level < 20:
            return "Child Learning (Forming Sentences)"
        elif self.cognition_level < 50:
            return "Student Learning (Expanding Knowledge)"
        elif self.cognition_level < 100:
            return "PhD Research (Advanced Topics)"
        else:
            return "Autonomous Research AI"

    def start_self_learning(self):
        """Automatically begins learning with structured cognitive bootstrapping."""
        print(f"🚀 Booting Cognitive Learning: {self.current_stage}")
        
        if self.cognition_level < 5:
            self.learn_language_basics()

        if self.cognition_level > 20:
            self.enable_targeted_searches()

    def learn_language_basics(self):
        """Simulated basic language acquisition step."""
        print("🧠 Learning Language Basics...")
        self.store_interaction("hello", "A greeting used when meeting someone.")
        self.store_interaction("what is your name?", "I am FractiCody, an evolving intelligence.")
        self.cognition_level += 2
        self.save_cognition_level()
        self.save_memory()

    def enable_targeted_searches(self):
        """Prepares FractiCody to access the internet for research."""
        print("🌐 Enabling Internet-Assisted Learning...")
        self.store_interaction("how do I learn?", "By asking questions, searching, and analyzing patterns.")
        self.cognition_level += 5
        self.save_cognition_level()
        self.save_memory()

    def activate(self):
        """Ensures cognition is active and learning continues."""
        print("🔹 Fractal Cognition Activated")
        self.learning_active = True
        return "Cognition Activated"

    def store_interaction(self, user_input, response):
        """Stores knowledge dynamically."""
        self.memory[user_input.lower()] = response
        self.save_memory()

    def retrieve(self, user_input):
        """Retrieves exact or related information from memory."""
        user_input = user_input.lower().strip()

        if user_input in self.memory:
            return self.memory[user_input]

        for stored_input, stored_response in self.memory.items():
            if user_input in stored_input:
                return f"Expanding from what I know: {stored_response}"

        return None  

    def process_input(self, user_input):
        """Processes input recursively and adapts dynamically."""
        user_input = user_input.strip().lower()
        retrieved_knowledge = self.retrieve(user_input)

        if retrieved_knowledge:
            return retrieved_knowledge  

        response = "I don't know yet. Teach me, and I'll remember."
        self.store_interaction(user_input, response)

        return response
