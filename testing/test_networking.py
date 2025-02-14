"""
🧪 Test - FractiNet AI Communication
Unit tests for P2P networking, data exchange, and security.
"""
import unittest
from networking.p2p_network import P2PNetwork
from networking.data_exchange import DataExchange
from networking.fracti_security import FractiSecurity

class TestNetworking(unittest.TestCase):
    def test_p2p_network(self):
        network = P2PNetwork()
        network.add_node("AI-1")
        self.assertIn("AI-1", network.network.nodes)

    def test_data_exchange(self):
        exchange = DataExchange()
        transaction = exchange.create_transaction("AI-1", "AI-2", "Test Data")
        self.assertTrue("Transaction Created" in transaction)

    def test_fracti_security(self):
        security = FractiSecurity()
        self.assertTrue("✅ Transaction Approved" in security.validate_transaction({"sender": "AI-1", "receiver": "AI-2", "data": "Valid Data"}))

if __name__ == "__main__":
    unittest.main()
