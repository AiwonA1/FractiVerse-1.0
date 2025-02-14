"""
🧪 Test - FractiChain Blockchain Operations
Unit tests for AI blockchain transactions, token economy, and treasury.
"""
import unittest
from blockchain.fracti_blockchain import FractiBlockchain
from blockchain.fracti_tokens import FractiToken
from blockchain.fracti_treasury import FractiTreasury

class TestBlockchain(unittest.TestCase):
    def test_blockchain_creation(self):
        chain = FractiBlockchain()
        self.assertEqual(len(chain.chain), 1)  # Genesis block exists

    def test_token_transactions(self):
        tokens = FractiToken()
        tokens.create_tokens("AI-1", 100)
        self.assertEqual(tokens.check_balance("AI-1"), "💰 AI-1 Balance: 100 FTN")

    def test_treasury_distribution(self):
        treasury = FractiTreasury()
        self.assertTrue("🎉" in treasury.distribute_rewards("AI-1", 500))

if __name__ == "__main__":
    unittest.main()
