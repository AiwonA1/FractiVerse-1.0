"""
💰 FractiToken (FTN) - AI-Powered Tokenized Knowledge Economy
Manages token creation, transactions, and AI-driven economy.
"""
class FractiToken:
    def __init__(self):
        self.token_ledger = {}  # Stores AI token balances

    def create_tokens(self, user, amount):
        """Creates and assigns tokens to a user."""
        self.token_ledger[user] = self.token_ledger.get(user, 0) + amount
        return f"💎 {amount} FTN Created for {user}"

    def transfer_tokens(self, sender, receiver, amount):
        """Transfers tokens between AI users."""
        if self.token_ledger.get(sender, 0) >= amount:
            self.token_ledger[sender] -= amount
            self.token_ledger[receiver] = self.token_ledger.get(receiver, 0) + amount
            return f"🔄 {amount} FTN Transferred from {sender} to {receiver}"
        return "❌ Insufficient Tokens for Transfer"

    def check_balance(self, user):
        """Returns the balance of a user."""
        return f"💰 {user} Balance: {self.token_ledger.get(user, 0)} FTN"

if __name__ == "__main__":
    fractitoken = FractiToken()
    print(fractitoken.create_tokens("AI-Node-1", 100))
    print(fractitoken.transfer_tokens("AI-Node-1", "AI-Node-2", 50))
    print(fractitoken.check_balance("AI-Node-2"))
