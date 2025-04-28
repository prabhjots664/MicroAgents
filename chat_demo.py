"""Demo showcasing custom message store with chat agents and metadata tracking."""

from datetime import datetime
from microAgents.llm import LLM
from microAgents.core import MicroAgent, Tool, BaseMessageStore

class ChatMessageStore(BaseMessageStore):
    """Extended message store that tracks timestamps and metadata."""
    
    def __init__(self):
        super().__init__()
        self.timestamps = []  # Track when each message was added
        self.metadata = {}    # Store additional info like user_id, chat_id etc
        self.chat_statistics = {
            'total_messages': 0,
            'messages_by_role': {},
            'average_response_time': 0
        }
    
    def add_message(self, message: dict, **metadata) -> int:
        """Add a message with timestamp and metadata."""
        message_idx = super().add_message(message)
        current_time = datetime.now()
        
        # Track timestamp
        self.timestamps.append(current_time)
        
        # Store metadata
        self.metadata[message_idx] = {
            'timestamp': current_time,
            **metadata
        }
        
        # Update statistics
        self.chat_statistics['total_messages'] += 1
        role = message.get('role', 'unknown')
        self.chat_statistics['messages_by_role'][role] = \
            self.chat_statistics['messages_by_role'].get(role, 0) + 1
            
        return message_idx
    
    def get_message_metadata(self, index: int) -> dict:
        """Get metadata for a specific message."""
        return self.metadata.get(index, {})
    
    def get_chat_statistics(self) -> dict:
        """Get current chat statistics."""
        return self.chat_statistics.copy()

# Initialize LLM
chat_llm = LLM(
    base_url="https://api.hyperbolic.xyz/v1",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJrYW1hbHNpbmdoZ2FsbGFAZ21haWwuY29tIiwiaWF0IjoxNzM1MjI2ODIzfQ.1wZmIzTZUWLzr-uP7Qtib_kkXNZmH_yQtSn1lP9S2z0",
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_tokens=4000,
    temperature=0.8,
    top_p=0.9
)

# Define chat tools
def get_current_time() -> str:
    """Get current time in a readable format."""
    return datetime.now().strftime("%I:%M %p")

def format_greeting(name: str) -> str:
    """Format a personalized greeting."""
    return f"Hello {name}! How can I assist you today?"

# Create chat agent
chat_agent = MicroAgent(
    llm=chat_llm,
    prompt="""You are a friendly chat assistant. Use available tools to enhance the conversation.
Keep responses concise and engaging. When greeting users, use the format_greeting tool.
If asked about the time, use the get_current_time tool.""",
    toolsList=[
        Tool(description="Get current time", func=get_current_time),
        Tool(description="Format greeting for a user", func=format_greeting)
    ]
)

def main():
    # Create custom message store with metadata tracking
    message_store = ChatMessageStore()
    
    # Example chat session
    queries = [
        {"text": "Hi, I'm Alice", "metadata": {"user_id": "alice123", "session_id": "abc789"}},
        {"text": "What time is it?", "metadata": {"user_id": "alice123", "session_id": "abc789"}},
        {"text": "Thanks for your help!", "metadata": {"user_id": "alice123", "session_id": "abc789"}}
    ]
    
    for query in queries:
        print(f"\nUser: {query['text']}")
        
        # Add user message with metadata
        message_store.add_message(
            {'role': 'user', 'content': query['text']},
            **query['metadata']
        )
        
        # Get agent response
        response = chat_agent.execute_agent(query['text'], message_store)
        print(f"Assistant: {response}")
        
        # After each exchange, print some statistics
        stats = message_store.get_chat_statistics()
        print(f"\nChat Statistics:")
        print(f"Total messages: {stats['total_messages']}")
        print(f"Messages by role: {stats['messages_by_role']}")
        
        # Get metadata of last message
        last_idx = len(message_store.messages) - 1
        metadata = message_store.get_message_metadata(last_idx)
        print(f"Last message timestamp: {metadata['timestamp'].strftime('%I:%M:%S %p')}")

if __name__ == "__main__":
    main()