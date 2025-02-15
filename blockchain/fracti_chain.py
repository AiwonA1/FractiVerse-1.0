"""
🔗 FractiChain - AI-Driven Fractal Blockchain
Handles AI Memory Persistence, Decentralized Transactions, and Smart Contracts.
"""

import hashlib
import json
import time
from datetime import datetime
from fracti_tokens import FractiToken
from fracti_treasury import FractiTreasury

class FractiBlock:
    def __init__(self, index, previous_hash, timestamp, data, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """Computes the hash of the block."""
        block_string = f"{self.index}{self.previous_hash}{self.timestamp}{json.dumps(self.data)}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

class FractiChain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.mining_rewards = {}
        self.add_genesis_block()

    def add_genesis_block(self):
        """Creates the first block in the blockchain."""
        genesis_block = FractiBlock(0, "0", int(time.time()), {"message": "Genesis Block"})
        self.chain.append(genesis_block)

    def get_latest_block(self):
        """Returns the most recent block."""
        return self.chain[-1]

    def add_block(self, data):
        """Adds a new block to the blockchain."""
        previous_block = self.get_latest_block()
        new_block = FractiBlock(len(self.chain), previous_block.hash, int(time.time()), data)
        self.chain.append(new_block)

    def is_chain_valid(self):
        """Validates the blockchain integrity."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def add_transaction(self, sender, recipient, amount, transaction_type="transfer"):
        """Adds a transaction to the pending list."""
        transaction = {
            "sender": sender,
            "recipient": recipient,
            "amount": amount,
            "timestamp": int(time.time()),
            "type": transaction_type
        }
        self.pending_transactions.append(transaction)

    def process_transactions(self):
        """Processes pending transactions and mines them into a block."""
        if not self.pending_transactions:
            return "No transactions to process."

        new_block_data = {
            "transactions": self.pending_transactions,
            "processed_by": "FractiChain AI"
        }
        self.add_block(new_block_data)
        self.pending_transactions = []
        return f"✅ Block added with {len(new_block_data['transactions'])} transactions."

    def mine_fracti_tokens(self, miner_address):
        """Enables FractiMining, allowing users to contribute computing power for rewards."""
        reward = FractiToken.mint_tokens(miner_address, 10)  # Rewarding 10 FractiTokens per mining cycle
        return f"✅ {reward} FractiTokens mined and sent to {miner_address}."

    def execute_smart_contract(self, contract_data):
        """Executes an AI-governed smart contract."""
        contract_hash = hashlib.sha256(json.dumps(contract_data).encode()).hexdigest()
        contract_data["execution_timestamp"] = datetime.utcnow().isoformat()
        contract_data["contract_hash"] = contract_hash

        self.add_block({"smart_contract": contract_data})
        return f"✅ Smart contract executed successfully: {contract_hash}"

    def store_ai_knowledge(self, knowledge_data):
        """Stores AI-learned insights in the decentralized knowledge ledger."""
        knowledge_entry = {
            "knowledge_id": hashlib.sha256(knowledge_data.encode()).hexdigest(),
            "content": knowledge_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.add_block({"ai_knowledge": knowledge_entry})
        return f"✅ AI knowledge stored with ID: {knowledge_entry['knowledge_id']}."

    def validate_transactions(self):
        """Implements adaptive consensus mechanism to validate transactions dynamically."""
        validated_transactions = []
        for tx in self.pending_transactions:
            if tx["amount"] > 0:  # Example validation
                validated_transactions.append(tx)

        self.pending_transactions = validated_transactions
        return f"✅ {len(validated_transactions)} transactions validated."

# Example Usage
if __name__ == "__main__":
    fracti_chain = FractiChain()
    
    fracti_chain.add_transaction("UserA", "UserB", 100)
    fracti_chain.process_transactions()
    
    fracti_chain.mine_fracti_tokens("Miner123")
    fracti_chain.execute_smart_contract({"contract_type": "data_storage", "details": "Store AI-generated insights"})
    
    fracti_chain.store_ai_knowledge("FractiCody has successfully processed recursive cognition optimization.")

    print("✅ FractiChain Initialized & Running!")
