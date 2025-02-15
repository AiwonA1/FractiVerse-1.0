"""
🧠 FractiChain - AI Blockchain for Fractal Memory Persistence
Implements Fractal Memory Ledger (FML), Recursive Consensus, and Cognitive Hash Mapping.
"""

import hashlib
import time
import json
from collections import deque

class FractiBlock:
    def __init__(self, index, previous_hash, transactions, timestamp, nonce=0):
        """Creates a new block for the FractiChain blockchain."""
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """Generates the block hash using Cognitive Hash Mapping (CHM)."""
        block_string = json.dumps({
            "index": self.index,
            "previous_hash": self.previous_hash,
            "transactions": self.transactions,
            "timestamp": self.timestamp,
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class FractiBlockchain:
    def __init__(self):
        """Initializes FractiChain with a genesis block."""
        self.chain = []
        self.memory_pool = deque()  # Temporary AI transaction pool
        self.create_genesis_block()

    def create_genesis_block(self):
        """Creates the first block in the blockchain (Genesis Block)."""
        genesis_block = FractiBlock(0, "0", ["Genesis AI Transaction"], time.time())
        self.chain.append(genesis_block)
        print("🟢 FractiChain Genesis Block Created")

    def add_block(self, transactions):
        """Adds a new AI transaction block to the chain."""
        last_block = self.chain[-1]
        new_block = FractiBlock(len(self.chain), last_block.hash, transactions, time.time())
        self.chain.append(new_block)
        print(f"🔗 New AI Memory Block Added - Index: {new_block.index}")

    def validate_chain(self):
        """Validates the blockchain using Recursive Consensus Mechanism (RCM)."""
        for i in range(1, len(self.chain)):
            prev_block = self.chain[i - 1]
            curr_block = self.chain[i]

            # Ensure hash integrity
            if curr_block.previous_hash != prev_block.hash:
                return False
            if curr_block.hash != curr_block.calculate_hash():
                return False
        return True

    def store_ai_memory(self, memory_data):
        """Processes AI memory data using Zero-Entropy Pruning (ZEP)."""
        if len(self.memory_pool) > 10:  # Threshold for optimization
            self.optimize_memory()
        self.memory_pool.append(memory_data)
        print(f"📝 AI Memory Stored: {memory_data}")

    def optimize_memory(self):
        """Prunes redundant AI data using Zero-Entropy Pruning (ZEP)."""
        unique_memories = set(self.memory_pool)  # Remove duplicates
        self.memory_pool = deque(unique_memories)
        print("♻️ AI Memory Pool Optimized (Zero-Entropy Pruning)")

    def retrieve_memory(self, index):
        """Retrieves AI memory block data by index."""
        if 0 <= index < len(self.chain):
            return self.chain[index].transactions
        return "❌ Memory Not Found"

# Example Usage
if __name__ == "__main__":
    fractichain = FractiBlockchain()
    
    # Store AI Memory
    fractichain.store_ai_memory("AI Cognitive Event - Recursive Thought")
    fractichain.store_ai_memory("AI Cognitive Event - Learning Fractal Patterns")
    
    # Add a New Block
    fractichain.add_block(["Fractal AI Decision Processing", "Optimized Thought Projection"])

    # Retrieve Memory
    retrieved_data = fractichain.retrieve_memory(1)
    print(f"🔍 Retrieved AI Memory: {retrieved_data}")

    # Validate Chain Integrity
    is_valid = fractichain.validate_chain()
    print(f"✅ Blockchain Integrity: {is_valid}")
