"""
🔗 FractiChain - AI Memory & Decentralized Knowledge Ledger
Handles the blockchain infrastructure for AI knowledge storage and verification.
"""
import hashlib
import time

class FractiBlock:
    def __init__(self, index, transactions, previous_hash):
        self.index = index
        self.timestamp = time.time()
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.hash = self.generate_hash()

    def generate_hash(self):
        """Generates a unique hash for the block."""
        block_string = f"{self.index}{self.timestamp}{self.transactions}{self.previous_hash}".encode()
        return hashlib.sha256(block_string).hexdigest()

class FractiBlockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        """Creates the first block in the chain."""
        genesis_block = FractiBlock(0, "Genesis Block", "0" * 64)
        self.chain.append(genesis_block)

    def add_block(self, transactions):
        """Adds a new block to the chain."""
        previous_hash = self.chain[-1].hash
        new_block = FractiBlock(len(self.chain), transactions, previous_hash)
        self.chain.append(new_block)
        return f"✅ Block {new_block.index} added with hash {new_block.hash}"

    def validate_chain(self):
        """Validates the blockchain integrity."""
        for i in range(1, len(self.chain)):
            if self.chain[i].previous_hash != self.chain[i - 1].hash:
                return "❌ Blockchain Integrity Compromised!"
        return "✅ Blockchain Integrity Verified"

if __name__ == "__main__":
    fractichain = FractiBlockchain()
    print(fractichain.add_block("Unipixel Intelligence Update"))
    print(fractichain.validate_chain())
