import time

class FractiCognition:
    def __init__(self):
        self.memory = []
        self.cognition_level = 1.0  # Base cognition level
        self.learning_active = True  # Enables deep learning by default
        self.max_memory_size = 100  # Prevents memory overflow

    def store_interaction(self, user_input, response):
        """Stores interactions for recursive learning."""
        if len(self.memory) >= self.max_memory_size:
            self.memory.pop(0)  # Remove oldest interaction to free space
        self.memory.append({"input": user_input, "response": response, "timestamp": time.time()})

    def retrieve_last(self):
        """Retrieves the most recent stored interaction."""
        return self.memory[-1] if self.memory else None

    def process_input(self, user_input):
        """Recursive AI cognition - adapts based on past interactions."""
        command = user_input.lower().strip()

        # Toggle learning mode
        if command == "activate deep learning":
            self.learning_active = True
            return "✅ Deep learning mode activated."

        if command == "deactivate deep learning":
            self.learning_active = False
            return "⛔ Deep learning mode deactivated."

        last_interaction = self.retrieve_last()

        if last_interaction:
            past_input = last_interaction["input"]
            past_response = last_interaction["response"]
            response = f"Based on our last conversation ({past_input}), I learned: {past_response}"
        else:
            response = "This is a new input. I'm analyzing and adapting..."

        # Recursive scaling: cognition level increases with each interaction
        self.cognition_level += 0.1
        response = f"[Cognition Level {self.cognition_level:.2f}] {response}"

        # If deep learning is active, extend reasoning
        if self.learning_active:
            response += " 🔄 Deep Learning Active."

        # Store learning data
        self.store_interaction(user_input, response)

        return response
