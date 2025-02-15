"""
📢 FractiAdmin - Unified Messaging System
Handles AI-driven system-wide announcements & user messaging.
"""

import time
from networking.p2p_network import P2PNetwork
from blockchain.fracti_blockchain import FractiChain

class BroadcastMessaging:
    def __init__(self):
        self.network = P2PNetwork()
        self.blockchain = FractiChain()
        self.message_queue = []

    def send_system_broadcast(self, message):
        """Sends a high-priority system-wide announcement."""
        formatted_message = f"🚨 SYSTEM ALERT: {message}"
        self.message_queue.append(formatted_message)
        self.network.broadcast(formatted_message)
        self.blockchain.log_message(formatted_message)
        print(f"✅ Broadcast sent: {formatted_message}")

    def send_user_message(self, sender, recipient, message):
        """Facilitates AI-managed user-to-user messaging."""
        formatted_message = f"📩 From {sender} to {recipient}: {message}"
        self.network.direct_message(sender, recipient, formatted_message)
        self.blockchain.log_message(formatted_message)
        print(f"📨 Message sent: {formatted_message}")

    def process_message_queue(self):
        """Processes and prioritizes queued messages."""
        while self.message_queue:
            message = self.message_queue.pop(0)
            print(f"🔄 Processing queued message: {message}")
            time.sleep(1)  # Simulated processing delay

if __name__ == "__main__":
    messaging_system = BroadcastMessaging()
    messaging_system.send_system_broadcast("🚀 FractiCody 1.0 is now fully live!")
    messaging_system.send_user_message("Admin", "All Users", "Welcome to the FractiVerse!")
    messaging_system.process_message_queue()
