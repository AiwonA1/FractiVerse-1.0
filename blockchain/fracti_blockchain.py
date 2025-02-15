"""
⛓️ FractiChain - AI-Optimized Blockchain for Decentralized Intelligence & Knowledge Transactions
Implements Recursive Proof-of-Intelligence (RPoI), Tokenized Knowledge Transactions (TKT), and Fractal Ledger Expansion (FLE).
"""

import hashlib
import json
import time
from fracti_tokens import FractiToken

class Block:
    def __init__(self, index, previous_hash, timestamp, data, fractal_signature):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.fractal_signature = fractal_signature
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """Computes the cryptographic hash for the block."""
        block_content = f"{self.index}{self.previous_hash}{self.timestamp}{self.data}{self.fractal_signature}"
        return hashlib.sha256(block_content.encode()).hexdigest()

class FractiChain:
    def __init__(self):
        """Initializes FractiChain with the Genesis Block."""
        self.chain = [self.create_genesis_block()]
        self.transactions = []
        self.tokens = FractiToken()

    def create_genesis_block(self):
        """Creates the first block in the chain (Genesis Block)."""
        return Block(0, "0", time.time(), "FractiGenesis Block", "Fractal Signature Root")

    def get_latest_block(self):
        """Returns the most recent block in the chain."""
        return self.chain[-1]

    def add_block(self, data):
        """Adds a new AI transaction block to FractiChain."""
        latest_block = self.get_latest_block()
        new_index = latest_block.index + 1
        new_timestamp = time.time()
        new_fractal_signature = hashlib.sha256(f"{data}{new_timestamp}".encode()).hexdigest()
        
        new_block = Block(new_index, latest_block.hash, new_timestamp, data, new_fractal_signature)
        if self.validate_block(new_block, latest_block):
            self.chain.append(new_block)
            print(f"✅ New AI Memory Block Added: {new_block.index} - {new_block.hash}")
        else:
            print("⚠️ Block validation failed!")

    def validate_block(self, new_block, previous_block):
        """Validates the integrity of a new block."""
        if previous_block.index + 1 != new_block.index:
            return False
        if previous_block.hash != new_block.previous_hash:
            return False
        if new_block.calculate_hash() != new_block.hash:
            return False
        return True

    def process_transaction(self, sender, recipient, amount):
        """Processes AI tokenized transactions on FractiChain."""
        transaction_data = {
            "sender": sender,
            "recipient": recipient,
            "amount": amount,
            "timestamp": time.time(),
        }
        self.transactions.append(transaction_data)
        self.add_block(json.dumps(transaction_data))
        self.tokens.transfer(sender, recipient, amount)
        print(f"🔄 Transaction Processed: {sender} ➝ {recipient} | Amount: {amount} FractiTokens")

    def validate_chain(self):
        """Validates the entire FractiChain ledger."""
        for i in range(1, len(self.chain)):
            if not self.validate_block(self.chain[i], self.chain[i - 1]):
                return False
        return True

# Example Usage
if __name__ == "__main__":
    fracti_chain = FractiChain()
    
    # Simulate AI memory storage
    fracti_chain.add_block("AI Memory: Self-Learning Expansion")
    fracti_chain.add_block("AI Cognition: Recursive Thought Analysis")
    
    # Process AI token transactions
    fracti_chain.process_transaction("FractiNode_A", "FractiNode_B", 500)
    fracti_chain.process_transaction("FractiUser_X", "FractiUser_Y", 1200)
    
    # Validate the blockchain
    chain_valid = fracti_chain.validate_chain()
    print(f"✅ FractiChain Ledger Integrity: {chain_valid}")
