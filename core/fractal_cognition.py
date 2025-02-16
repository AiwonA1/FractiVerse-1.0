import time
import sys
import os
import json

class FractiCognition:
    def __init__(self):
        self.memory = {}  # Stores learned facts
        self.cognition_level = 1.0  # Tracks cognitive expansion
        self.learning_active = True  # Enables continuous learning

    def store_interaction(self, user_input, response):
        """Stores knowledge dynamically."""
        self.memory[user_input.lower()] = response

    def retrieve(self, user_input):
        """Retrieves exact or related information from memory."""
        user_input = user_input.lower().strip()

        # Direct recall if exact match exists
        if user_input in self.memory:
            return self.memory[user_input]

        # Find closest related knowledge
        for stored_input, stored_response in self.memory.items():
            if user_input in stored_input:
                return f"Expanding from what I know: {stored_response}"

        # If no match, acknowledge and prepare to learn
        return None  

    def process_input(self, user_input):
        """Processes input recursively and adapts dynamically."""
        user_input = user_input.strip().lower()
        retrieved_knowledge = self.retrieve(user_input)

        if retrieved_knowledge:
            return retrieved_knowledge  # Return known info or closest match

        # If new input, encourage teaching
        response = f"I don't know yet. Teach me, and I'll remember."
        self.store_interaction(user_input, response)

        return response
