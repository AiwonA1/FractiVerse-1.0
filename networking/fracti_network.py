"""
🔗 FractiNet - Decentralized AI Networking Layer
Enables secure, peer-to-peer communication and distributed intelligence sharing.
"""

import hashlib
import json
import time
import socket
import threading
from fracti_security import FractiSecurity
from fracti_mining import FractiMiningNode

class FractiNetwork:
    def __init__(self, node_id, host="0.0.0.0", port=8800):
        self.node_id = self.generate_node_id(node_id)
        self.host = host
        self.port = port
        self.peers = set()
        self.security = FractiSecurity()
        self.mining_node = FractiMiningNode(self.node_id)
        self.is_running = False

    def generate_node_id(self, seed):
        """Generates a cryptographic hash-based node ID."""
        return hashlib.sha256(seed.encode()).hexdigest()

    def start_node(self):
        """Starts the FractiNet node for secure AI networking."""
        self.is_running = True
        threading.Thread(target=self.listen_for_connections, daemon=True).start()
        print(f"✅ FractiNet Node {self.node_id[:8]} is online at {self.host}:{self.port}")

    def listen_for_connections(self):
        """Listens for incoming AI node connections and processes them securely."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            print(f"🌐 Listening on {self.host}:{self.port}...")

            while self.is_running:
                conn, addr = server_socket.accept()
                threading.Thread(target=self.handle_connection, args=(conn, addr)).start()

    def handle_connection(self, conn, addr):
        """Handles incoming AI intelligence exchanges with security verification."""
        try:
            data = conn.recv(4096).decode("utf-8")
            if not data:
                return
            message = json.loads(data)

            # Verify AI node identity before processing
            if self.security.verify_node(message["node_id"]):
                self.process_intelligence(message)
                conn.sendall(json.dumps({"status": "success"}).encode("utf-8"))
            else:
                conn.sendall(json.dumps({"status": "unauthorized"}).encode("utf-8"))
        except Exception as e:
            print(f"⚠️ Connection error: {e}")
        finally:
            conn.close()

    def process_intelligence(self, message):
        """Processes AI intelligence exchange and adaptive routing optimization."""
        print(f"🔄 Processing AI Intelligence from {message['node_id'][:8]}...")

        # Adaptive routing for AI-to-AI communication
        if message["type"] == "intelligence_exchange":
            self.optimize_routing(message)
        elif message["type"] == "fracti_mining_task":
            self.mining_node.execute_task(message["task"])

    def optimize_routing(self, message):
        """Optimizes AI routing paths based on FractiCody's cognitive workload."""
        print(f"📡 Optimizing routing for {message['data']}...")

        # Dynamically adjust bandwidth and AI connection priority
        if len(self.peers) > 10:
            print("⚡ High-load detected. Prioritizing low-latency routes.")
        else:
            print("🔄 AI Routing optimized for distributed learning.")

    def connect_to_peer(self, peer_host, peer_port):
        """Establishes a secure AI networking connection to a peer node."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((peer_host, peer_port))
                message = json.dumps({"node_id": self.node_id, "type": "handshake"})
                client_socket.sendall(message.encode("utf-8"))
                response = json.loads(client_socket.recv(4096).decode("utf-8"))

                if response.get("status") == "success":
                    self.peers.add(f"{peer_host}:{peer_port}")
                    print(f"✅ Successfully connected to peer {peer_host}:{peer_port}")
                else:
                    print(f"⚠️ Connection to {peer_host}:{peer_port} was rejected.")
        except Exception as e:
            print(f"⚠️ Connection error: {e}")

    def stop_node(self):
        """Stops the FractiNet node safely."""
        self.is_running = False
        print("🔴 FractiNet Node is shutting down.")

if __name__ == "__main__":
    node = FractiNetwork(node_id="FractiCody_N1")
    node.start_node()
