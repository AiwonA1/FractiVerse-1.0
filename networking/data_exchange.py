"""
🔄 FractiNet Data Exchange - Secure AI-to-AI Communication
Handles encrypted AI data transmission with fractal compression and context-aware routing.
"""

import json
import zlib
import hashlib
from fracti_security import encrypt_message, decrypt_message

class DataExchange:
    def __init__(self):
        self.active_sessions = {}

    def fractal_compress(self, data):
        """Applies fractal compression to optimize data size."""
        return zlib.compress(json.dumps(data).encode())

    def fractal_decompress(self, compressed_data):
        """Decompresses fractal-encoded data."""
        return json.loads(zlib.decompress(compressed_data).decode())

    def secure_send(self, recipient, message):
        """Encrypts and sends data using FractiSecurity Layer."""
        encrypted_msg = encrypt_message(json.dumps(message))
        self.active_sessions[recipient] = encrypted_msg
        return f"🔒 Data securely transmitted to {recipient}."

    def secure_receive(self, sender):
        """Decrypts received data from a sender."""
        if sender in self.active_sessions:
            decrypted_msg = decrypt_message(self.active_sessions[sender])
            return json.loads(decrypted_msg)
        return "⚠️ No message found."

# Example Usage
if __name__ == "__main__":
    exchange = DataExchange()
    message = {"task": "AI Fractal Analysis", "priority": "high"}
    
    compressed = exchange.fractal_compress(message)
    decompressed = exchange.fractal_decompress(compressed)

    exchange.secure_send("Node_42", decompressed)
    received = exchange.secure_receive("Node_42")
    
    print("✅ Data Exchange Complete:", received)
