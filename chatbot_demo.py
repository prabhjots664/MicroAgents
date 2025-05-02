from microAgents.llm import LLM
from microAgents.core import MicroAgent, Tool, BaseMessageStore
import datetime

# Initialize LLM
chat_llm = LLM(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    model="qwen/qwen3-14b:free",
    max_tokens=4000,
    temperature=0.8,
    top_p=0.9
)

# Global in-memory databases
orders_db = {
    "1001": {"items": ["Large Pepperoni"], "status": "Delivered"},
    "1002": {"items": ["Medium Veggie"], "status": "Preparing"}
}

concerns_db = {}
next_concern_id = 1

# Define tools
def order_details(order_id: str) -> dict:
    """Returns details for a specific order"""
    return orders_db.get(order_id, {"error": "Order not found"})

def order_history() -> list:
    """Returns order history for the single user"""
    return list(orders_db.values())

def create_concern(description: str) -> dict:
    """Creates and stores a new customer concern for the single user"""
    global next_concern_id
    concern_id = f"CON-{next_concern_id}"
    concerns_db[concern_id] = {
        "id": concern_id,
        "description": description,
        "status": "Open",
        "timestamp": datetime.datetime.now().isoformat()
    }
    next_concern_id += 1
    return concerns_db[concern_id]

def get_concern_details(concern_id: str) -> dict:
    """Returns details for a specific concern"""
    return concerns_db.get(concern_id, {"error": "Concern not found"})

# Create agents
order_agent = MicroAgent(
    llm=chat_llm,
    prompt="""You are an order management assistant. Handle order details and history.""",
    toolsList=[
        Tool(description="Returns order details", func=order_details),
        Tool(description="Returns order history", func=order_history)
    ]
)

concern_agent = MicroAgent(
    llm=chat_llm,
    prompt="""You are a customer concern assistant. Handle customer issues and concerns.""",
    toolsList=[
        Tool(description="Creates a new concern", func=create_concern),
        Tool(description="Returns concern details", func=get_concern_details)
    ]
)

class Orchestrator(MicroAgent):
    def __init__(self):
        super().__init__(
            llm=chat_llm,
            prompt="""You are a chat orchestrator. Route messages to appropriate agents.

1. For order-related queries, output exactly: ORDER_AGENT NEEDED
2. For concern-related queries, output exactly: CONCERN_AGENT NEEDED
3. Otherwise respond like a generic chat assistant
4. Do not use any tools yourself.""",
            toolsList=[]
        )
        self.order_agent = order_agent
        self.concern_agent = concern_agent

    def execute_agent(self, query: str, message_store: BaseMessageStore) -> str:
        """Handle full query flow through orchestrator."""
        # print(f"\nDebug: Orchestrator analyzing query: {query}")

        # Get initial analysis from orchestrator
        analysis = super().execute_agent(query, message_store)
        # print(f"Debug: Orchestrator analysis result: {analysis}")

        if "ORDER_AGENT NEEDED" in analysis:
            # print("Debug: Routing to Order Agent")
            result = self.order_agent.execute_agent(query, message_store)
            # print(f"Debug: Order Agent result: {result}")
            return self._format_result("Order Agent", result)
        elif "CONCERN_AGENT NEEDED" in analysis:
            # print("Debug: Routing to Concern Agent")
            result = self.concern_agent.execute_agent(query, message_store)
            # print(f"Debug: Concern Agent result: {result}")
            return self._format_result("Concern Agent", result)
        else:
            return analysis

    def _format_result(self, agent_name: str, result: str) -> str:
        """Format the final result from an agent."""
        return f"Orchestrator: Result from {agent_name}:\n{result}"

if __name__ == "__main__":
    message_store = BaseMessageStore()
    orchestrator = Orchestrator()

    print("Welcome to Dominos Pizza Chat Support!")
    print("I can help with order details, order history, or concerns/issues.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break

        response = orchestrator.execute_agent(user_input, message_store)
        print(f"Agent: {response}")
        # print("Debug: Orchestrator routing complete")
