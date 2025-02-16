import time
import json
import os

class FractiCognition:
    def __init__(self, memory_file="fracti_memory.json"):
        self.memory = []  # Stores interactions
        self.memory_file = memory_file  # Persistent storage file
        self.cognition_level = 1.0  # Starts at base cognition
        self.learning_active = True  # Enables deep learning
        self.load_memory()  # Load existing memory on initialization

    def load_memory(self):
        """Loads stored memory from file if available."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as file:
                    self.memory = json.load(file)
            except json.JSONDecodeError:
                self.memory = []  # Reset memory if file is corrupted
        
    def save_memory(self):
        """Saves memory to file for persistence."""
        with open(self.memory_file, "w") as file:
            json.dump(self.memory, file, indent=4)

    def store_interaction(self, user_input, response):
        """Stores interactions for recursive learning."""
        interaction = {"input": user_input, "response": response, "timestamp": time.time()}
        self.memory.append(interaction)
        self.save_memory()  # Persist memory

    def retrieve_last(self):
        """Retrieves the most recent stored interaction."""
        return self.memory[-1] if self.memory else None

    def process_input(self, user_input):
        """Recursive AI cognition - adapts based on past interactions."""
        last_interaction = self.retrieve_last()
        
        if last_interaction:
            past_input = last_interaction["input"]
            past_response = last_interaction["response"]
            response = f"[Cognition Level {self.cognition_level:.2f}] Based on our last conversation ({past_input}), I learned: {past_response}"
        else:
            response = f"[Cognition Level {self.cognition_level:.2f}] This is a new input. I'm forming my initial understanding..."
        
        # Increase cognition level
        self.cognition_level += 0.1
        
        # If deep learning is on, expand reasoning
        if self.learning_active:
            response += " 🔄 Deep Learning Active."
        
        # Store learning data
        self.store_interaction(user_input, response)
        
        return response
