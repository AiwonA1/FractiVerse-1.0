import time

class FractiCognition:
    def __init__(self):
        self.memory = []
        self.cognition_level = 1.0  # Starts at base cognition
        self.learning_active = True  # Enables deep learning

    def store_interaction(self, user_input, response):
        """Stores interactions for recursive learning."""
        self.memory.append({"input": user_input, "response": response, "timestamp": time.time()})

    def retrieve_last(self):
        """Retrieves the most recent stored interaction."""
        return self.memory[-1] if self.memory else None

    def process_input(self, user_input):
        """Recursive AI cognition - adapts based on past interactions."""
        last_interaction = self.retrieve_last()
        
        if last_interaction:
            past_input, past_response = last_interaction["input"], last_interaction["response"]
            response = f"Based on our last conversation ({past_input}), I learned: {past_response}"
        else:
            response = "This is new input. I'm analyzing..."

        # Recursive scaling: The more interactions, the more depth of response
        self.cognition_level += 0.1
        response = f"[Cognition Level {self.cognition_level:.2f}] {response}"

        # If deep learning is on, expand reasoning
        if self.learning_active:
            response += " 🔄 Deep Learning Active."

        # Store learning data
        self.store_interaction(user_input, response)

        return response
