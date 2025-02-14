"""
🌐 FractiNet P2P AI Network - Decentralized Communication Layer
Handles peer-to-peer AI networking, data flow, and distributed intelligence.
"""
import networkx as nx

class P2PNetwork:
    def __init__(self):
        self.network = nx.Graph()  # Represents AI node connections

    def add_node(self, node_id):
        """Adds an AI node to the decentralized network."""
        self.network.add_node(node_id)
        return f"🔹 Node {node_id} added to FractiNet"

    def connect_nodes(self, node1, node2, weight=1.0):
        """Establishes a connection between two AI nodes."""
        if node1 in self.network.nodes and node2 in self.network.nodes:
            self.network.add_edge(node1, node2, weight=weight)
            return f"🔗 Connected {node1} ↔ {node2} (Weight: {weight})"
        return "⚠️ One or both nodes do not exist."

    def display_network(self):
        """Prints current network structure."""
        return f"🌐 FractiNet Structure: {self.network.edges}"

if __name__ == "__main__":
    net = P2PNetwork()
    print(net.add_node("AI-Node-1"))
    print(net.add_node("AI-Node-2"))
    print(net.connect_nodes("AI-Node-1", "AI-Node-2", weight=0.9))
    print(net.display_network())
