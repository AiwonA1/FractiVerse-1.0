"""
🔗 FractiChain - Decentralized AI Memory & Tokenization System
Handles AI-powered blockchain transactions, tokenized knowledge, and FractiMining.
"""

import hashlib
import json
import time
from typing import List, Dict
from fracti_tokens import FractiToken
from fracti_treasury import FractiTreasury

class Block:
    def __init__(self, index, previous_hash, transactions, timestamp=None):
        self.index = index
        self.timestamp = timestamp or time.time()
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.compute_hash()

    def compute_hash(self):
        """Creates SHA-256 hash of the block content."""
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class FractiBlockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict] = []
        self.create_genesis_block()
        self.fracti_treasury = FractiTreasury()

    def create_genesis_block(self):
        """Creates the first block in the blockchain."""
        genesis_block = Block(index=0, previous_hash="0", transactions=[])
        self.chain.append(genesis_block)

    def add_transaction(self, sender, recipient, amount, data=""):
        """Adds a transaction to the pending transactions list."""
        transaction = {
            "sender": sender,
            "recipient": recipient,
            "amount": amount,
            "data": data,
            "timestamp": time.time()
        }
        self.pending_transactions.append(transaction)

    def mine_block(self, miner_address):
        """Mines a new block and adds it to the chain."""
        if not self.pending_transactions:
            return False

        last_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            previous_hash=last_block.hash,
            transactions=self.pending_transactions
        )

        self.chain.append(new_block)
        self.pending_transactions = []

        # Reward miner with FractiTokens
        self.fracti_treasury.issue_tokens(miner_address, 10)
        return new_block

    def is_valid_chain(self):
        """Verifies the integrity of the blockchain."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if current.previous_hash != previous.hash:
                return False
            if current.hash != current.compute_hash():
                return False
        return True

    def get_last_block(self):
        """Retrieves the latest block in the chain."""
        return self.chain[-1]

# 🔹 Example Usage
if __name__ == "__main__":
    fractichain = FractiBlockchain()

    # Add sample transactions
    fractichain.add_transaction("User1", "User2", 50, "AI knowledge transfer")
    fractichain.add_transaction("User3", "User4", 30, "FractiMining Reward")

    # Mine a block
    new_block = fractichain.mine_block("MinerNode1")

    # Display blockchain status
    print(f"✅ Blockchain Length: {len(fractichain.chain)}")
    print(f"🔗 Last Block Hash: {fractichain.get_last_block().hash}")
    print(f"💰 Treasury Balance: {fractichain.fracti_treasury.get_balance('MinerNode1')}")
