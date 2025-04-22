"""Simple shared message state management for microAgents framework."""

class MessageStore:
    def __init__(self):
        self.messages = []
        
    def add_message(self, message: dict) -> int:
        """Add a message to the store and return its index."""
        self.messages.append(message)
        return len(self.messages) - 1
        
    def get_messages(self) -> list:
        """Get a copy of all messages to prevent modification."""
        return self.messages.copy()