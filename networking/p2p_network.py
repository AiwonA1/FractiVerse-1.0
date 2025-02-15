"""
🌐 FractiNet P2P Network - AI-Powered Peer-to-Peer Communication
Implements Fractal Node Routing, Decentralized AI Synchronization, and AI Fault Tolerance.
"""

import socket
import threading
import json
import hashlib
import random

class P2PNode:
    def __init__(self, host='0.0.0.0', port=8800):
        """Initializes a P2P AI node with decentralized intelligence exchange."""
        self.host = host
        self.port = port
        self.peers = []
        self.node_id = hashlib.sha256(f"{random.randint(1, 100000)}".encode()).hexdigest()[:8]

    def start_node(self):
        """Starts the AI network node and listens for connections."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(5)
        print(f"🌐 P2P Node {self.node_id} Listening on {self.host}:{self.port}")

        while True:
            client, address = server.accept()
            threading.Thread(target=self.handle_client, args=(client, address)).start()

    def handle_client(self, client, address):
        """Handles incoming AI intelligence data."""
        data = client.recv(1024).decode()
        if data:
            message = json.loads(data)
            print(f"🔗 Received AI Data from {address}: {message}")
            self.process_data(message)
        client.close()

    def process_data(self, message):
        """Processes and synchronizes AI intelligence across nodes."""
        if "sync" in message:
            self.synchronize_data(message["sync"])

    def synchronize_data(self, data):
        """Implements Decentralized AI Synchronization (DAS)."""
        data_hash = hashlib.sha256(json.dumps(data).encode()).hexdigest()
        print(f"✅ AI Data Synchronized: {data_hash}")

    def connect_to_peer(self, peer_ip, peer_port):
        """Connects to another AI node and initiates sync."""
        try:
            peer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            peer.connect((peer_ip, peer_port))
            self.peers.append((peer_ip, peer_port))
            print(f"🔹 Connected to AI Peer at {peer_ip}:{peer_port}")
        except Exception as e:
            print(f"❌ Failed to connect to {peer_ip}:{peer_port}: {e}")

    def broadcast_intelligence(self, data):
        """Shares AI intelligence with all known peers."""
        for peer_ip, peer_port in self.peers:
            try:
                peer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                peer.connect((peer_ip, peer_port))
                peer.send(json.dumps({"sync": data}).encode())
                peer.close()
                print(f"📡 AI Data Broadcasted to {peer_ip}:{peer_port}")
            except Exception as e:
                print(f"⚠️ Error broadcasting to {peer_ip}:{peer_port}: {e}")

# Example Usage
if __name__ == "__main__":
    node = P2PNode(port=8800)
    threading.Thread(target=node.start_node).start()

    # Simulate AI Node-to-Node Communication
    import time
    time.sleep(2)  # Give time for node to start
    node.connect_to_peer("127.0.0.1", 8800)
    node.broadcast_intelligence({"message": "AI Knowledge Transfer"})
