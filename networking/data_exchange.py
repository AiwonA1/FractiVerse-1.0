"""
🔄 AI Data Exchange - FractiChain-Enabled Knowledge Transactions
Handles secure AI-to-AI data flow, ensuring efficient recursive knowledge sharing.
"""
import hashlib

class DataExchange:
    def __init__(self):
        self.ledger = []  # Stores knowledge transactions

    def create_transaction(self, sender, receiver, data):
        """Creates a knowledge transaction between AI entities."""
        transaction = {
            "sender": sender,
            "receiver": receiver,
            "data": data,
            "hash": self.generate_hash(sender, receiver, data)
        }
        self.ledger.append(transaction)
        return f"📡 Transaction Created: {transaction}"

    def generate_hash(self, sender, receiver, data):
        """Generates a secure hash for the transaction."""
        transaction_string = f"{sender}{receiver}{data}".encode()
        return hashlib.sha256(transaction_string).hexdigest()

    def display_ledger(self):
        """Displays all AI knowledge transactions."""
        return self.ledger

if __name__ == "__main__":
    exchange = DataExchange()
    print(exchange.create_transaction("AI-Node-1", "AI-Node-2", "Fractal Intelligence Update"))
    print(exchange.display_ledger())
