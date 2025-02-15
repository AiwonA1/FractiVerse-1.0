"""
💰 FractiTokens - AI-Powered Knowledge Economy & Tokenized Intelligence Transactions
Manages FractiToken issuance, transactions, and FractiMining rewards.
"""

import hashlib
import time

class FractiToken:
    def __init__(self):
        """Initializes the FractiToken ledger and sets the initial supply."""
        self.token_ledger = {}
        self.total_supply = 1_000_000  # Initial supply
        self.reward_rate = 10  # Tokens earned per AI knowledge contribution

    def mint_tokens(self, recipient, amount):
        """Mints new FractiTokens for AI-generated knowledge and user contributions."""
        if recipient not in self.token_ledger:
            self.token_ledger[recipient] = 0
        self.token_ledger[recipient] += amount
        self.total_supply += amount
        print(f"🟢 Minted {amount} FractiTokens to {recipient}")

    def transfer(self, sender, recipient, amount):
        """Processes token transactions between AI nodes and users."""
        if sender not in self.token_ledger or self.token_ledger[sender] < amount:
            print(f"⚠️ Transfer failed: Insufficient balance for {sender}")
            return False

        if recipient not in self.token_ledger:
            self.token_ledger[recipient] = 0

        self.token_ledger[sender] -= amount
        self.token_ledger[recipient] += amount
        print(f"🔄 Transferred {amount} FractiTokens from {sender} ➝ {recipient}")
        return True

    def get_balance(self, account):
        """Returns the token balance of a given account."""
        return self.token_ledger.get(account, 0)

    def process_fracti_mining(self, miner, computing_power):
        """Rewards users for contributing computing power to FractiCody via FractiMining."""
        earned_tokens = int(computing_power * self.reward_rate)
        self.mint_tokens(miner, earned_tokens)
        print(f"💎 FractiMining Reward: {miner} earned {earned_tokens} FractiTokens for contributing {computing_power} CRUs.")

    def execute_smart_contract(self, sender, recipient, amount, contract_conditions):
        """Executes AI-powered smart contracts based on tokenized transactions."""
        if contract_conditions():
            return self.transfer(sender, recipient, amount)
        print(f"❌ Smart contract execution failed for {sender} -> {recipient}")
        return False

# Example Usage
if __name__ == "__main__":
    fracti_token = FractiToken()

    # Simulate token minting for AI knowledge contributions
    fracti_token.mint_tokens("FractiUser_1", 500)
    fracti_token.mint_tokens("FractiNode_A", 1200)

    # Process token transactions
    fracti_token.transfer("FractiUser_1", "FractiUser_2", 200)

    # Simulate FractiMining reward system
    fracti_token.process_fracti_mining("FractiMiner_X", 50)  # 50 CRUs contributed

    # Execute an AI smart contract
    fracti_token.execute_smart_contract(
        "FractiUser_1", "FractiService_Y", 300, lambda: True  # Contract always passes
    )

    # Display balances
    print(f"🔹 FractiUser_1 Balance: {fracti_token.get_balance('FractiUser_1')} FractiTokens")
    print(f"🔹 FractiUser_2 Balance: {fracti_token.get_balance('FractiUser_2')} FractiTokens")
