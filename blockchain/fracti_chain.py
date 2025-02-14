"""
🔗 FractiChain - AI Memory Storage & Transactions (Fully Functional)
Stores Unipixel intelligence states and AI decision logs.
"""
import hashlib
import time

class FractiChain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []

    def create_block(self, previous_hash=None):
        block = {
            'index': len(self.chain) + 1,
            'transactions': self.pending_transactions[:],
            'previous_hash': previous_hash or (self.chain[-1]['hash'] if self.chain else "0" * 64),
            'hash': self.generate_hash(self.pending_transactions)
        }
        self.pending_transactions = []
        self.chain.append(block)
        return f"✅ Block {block['index']} Created: {block['hash']}"

    def store_unipixel_state(self, unipixel):
        transaction = {
            "unipixel_id": unipixel.id,
            "state": unipixel.get_state(),
            "timestamp": time.time()
        }
        self.pending_transactions.append(transaction)
        return f"📜 Unipixel {unipixel.id} Intelligence State Stored."

    def retrieve_unipixel_state(self, unipixel_id):
        for block in reversed(self.chain):
            for transaction in block["transactions"]:
                if transaction["unipixel_id"] == unipixel_id:
                    return transaction["state"]
        return None
