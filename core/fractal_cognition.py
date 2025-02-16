import time

class FractiCognition:
    def __init__(self):
        self.memory = {}
        self.cognition_level = 1.0  # Starts at base cognition
        self.learning_active = True  # Enables deep learning

    def store_interaction(self, user_input, response):
        """Stores interactions for recursive learning."""
        self.memory[user_input.lower()] = response

    def retrieve_last(self):
        """Retrieves the most recent stored fact, if any."""
        return list(self.memory.items())[-1] if self.memory else None

    def process_input(self, user_input):
        """Recursive AI cognition - adapts based on past interactions."""
        user_input = user_input.strip().lower()

        # Direct recall of learned facts
        if user_input in self.memory:
            return self.memory[user_input]

        # Try to find related knowledge
        for stored_input, stored_response in self.memory.items():
            if user_input in stored_input:
                return f"Building on what I know: {stored_response}"

        # If it's new input, acknowledge and learn
        response = f"I don't know yet. Please teach me."
        self.store_interaction(user_input, response)

        return response
