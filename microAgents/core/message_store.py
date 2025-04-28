"""Base message store management for microAgents framework.

This module provides a base BaseMessageStore class that can be extended with additional
fields and functionality while maintaining compatibility with the core framework.
"""

class BaseMessageStore:
    """Base class for message storage and management.
    
    This class can be extended to add additional fields and functionality
    while maintaining compatibility with the core framework.
    
    Example:
        class CustomMessageStore(BaseMessageStore):
            def __init__(self):
                super().__init__()
                self.custom_field = []
                
            def custom_method(self):
                pass
    """
    
    def __init__(self):
        """Initialize base message store with messages list."""
        self.messages = []
        
    def add_message(self, message: dict) -> int:
        """Add a message to the store and return its index.
        
        Args:
            message (dict): Message dictionary with 'role' and 'content'
            
        Returns:
            int: Index of the newly added message
        """
        self.messages.append(message)
        return len(self.messages) - 1
        
    def get_messages(self) -> list:
        """Get a copy of all messages to prevent modification.
        
        Returns:
            list: Copy of all stored messages
        """
        return self.messages.copy()
        
    def clear_messages(self) -> None:
        """Clear all messages from the store."""
        self.messages.clear()
        
    def get_last_message(self) -> dict:
        """Get the last message in the store.
        
        Returns:
            dict: The last message or None if store is empty
        """
        return self.messages[-1] if self.messages else None