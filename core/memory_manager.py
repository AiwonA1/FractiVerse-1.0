"""
🧠 AI Memory Manager - FractiChain-Enabled Memory Storage
Handles long-term recursive memory and deep history tracking.
"""
from collections import deque

class MemoryManager:
    def __init__(self, max_size=100):
        self.memory = deque(maxlen=max_size)

    def store_memory(self, data):
        """Stores AI experiences and knowledge recursively."""
        self.memory.append(data)
        return f"📥 Memory Stored: {data}"

    def retrieve_memory(self, index=-1):
        """Retrieves past AI experiences."""
        if not self.memory:
            return "⚠️ No Memory Available"
        return f"📤 Retrieved Memory: {self.memory[index]}"

    def clear_memory(self):
        """Erases AI memory (only when necessary)."""
        self.memory.clear()
        return "🧹 AI Memory Cleared"

if __name__ == "__main__":
    memory_manager = MemoryManager()
    print(memory_manager.store_memory("AI learned recursion."))
    print(memory_manager.retrieve_memory())
    print(memory_manager.clear_memory())
