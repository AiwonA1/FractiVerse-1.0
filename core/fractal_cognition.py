import time
import json

class FractalCognition:
    def __init__(self):
        print("✅ Fractal Cognition Initialized")
        self.memory = {}
        self.cognition_level = 1.0
        self.learning_active = True

        # Ensure module initialization delay (helps with race conditions)
        time.sleep(1)

    def activate(self):
        """Activates cognition if it was paused"""
        print("🔹 Fractal Cognition Activated")
        self.learning_active = True
        return "Cognition Activated"

    def store_interaction(self, user_input, response):
        """Stores knowledge dynamically."""
        self.memory[user_input.lower()] = response

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

