"""
📢 Broadcast Messaging - System-Wide AI Directives & Communication
Allows administrators to send AI system-wide directives and intelligence updates.
"""
class BroadcastMessaging:
    def __init__(self):
        self.messages = []

    def send_message(self, message):
        """Broadcasts an administrative message to all AI nodes."""
        self.messages.append(message)
        return f"📡 Broadcast Sent: {message}"

    def list_messages(self):
        """Displays all sent administrative messages."""
        return f"📢 AI Directives: {self.messages if self.messages else 'No Messages Sent'}"

if __name__ == "__main__":
    messaging = BroadcastMessaging()
    print(messaging.send_message("Initiate Recursive Optimization Protocol"))
    print(messaging.list_messages())
