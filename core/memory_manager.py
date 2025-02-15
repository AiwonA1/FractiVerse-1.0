"""
🗂️ Memory Manager - FractiCody Long-Term Cognitive Memory System
Manages recursive knowledge retention, structured recall, and entropy-based pruning.
"""

import hashlib
import numpy as np
import random

class MemoryManager:
    def __init__(self):
        self.memory_store = {}
        self.entropy_threshold = 0.3  # Controls knowledge pruning sensitivity

    def generate_memory_id(self, data):
        """Generates a unique memory ID using SHA-256."""
        return hashlib.sha256(data.encode()).hexdigest()

    def store_memory(self, memory_id, data):
        """Stores structured knowledge fragments recursively."""
        if memory_id not in self.memory_store:
            self.memory_store[memory_id] = {"data": data, "entropy_score": self.calculate_entropy(data)}
        return f"🧠 Memory Stored: {memory_id[:8]}"

    def retrieve_memory(self, memory_id):
        """Retrieves stored knowledge if available."""
        return self.memory_store.get(memory_id, {}).get("data", "⚠️ Memory Not Found")

    def calculate_entropy(self, data):
        """Computes entropy level of stored knowledge to determine its retention priority."""
        symbol_counts = {char: data.count(char) for char in set(data)}
        probabilities = np.array(list(symbol_counts.values())) / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def prune_memory(self):
        """Automatically removes low-entropy memories to optimize cognition."""
        pruned = []
        for memory_id, entry in list(self.memory_store.items()):
            if entry["entropy_score"] < self.entropy_threshold:
                del self.memory_store[memory_id]
                pruned.append(memory_id[:8])
        return f"🧹 Pruned Memories: {', '.join(pruned) if pruned else 'None'}"

    def recursive_memory_indexing(self, memory_fragments, depth=0):
        """Recursively structures memories into fractal sequences for rapid retrieval."""
        if depth >= 3:
            return "🔄 Recursive Memory Structuring Complete"

        indexed_fragments = {self.generate_memory_id(frag): frag for frag in memory_fragments}
        self.memory_store.update(indexed_fragments)
        return self.recursive_memory_indexing(list(indexed_fragments.values()), depth + 1)

if __name__ == "__main__":
    memory = MemoryManager()
    sample_memory = "FractiCody AI Adaptive Learning"
    memory_id = memory.generate_memory_id(sample_memory)
    print(memory.store_memory(memory_id, sample_memory))
    print(memory.prune_memory())
    print(memory.recursive_memory_indexing([sample_memory]))
