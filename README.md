# microAgents Framework

A lightweight LLM orchestration framework for building Multi-Agent AI systems. The framework provides an easy way to create and orchestrate multiple AI agents with XML-style tool calls.

## Key Features

🚀 **Universal Tool Calling Support**
- Works with ANY LLM API that follows OpenAI-compatible format
- **Unique Feature**: Enables function/tool calling even with models that don't natively support it
- XML-based tool calling format that's intuitive and human-readable

## Framework Comparison

| Framework    | Package Size | Core Features | Key Differentiator | Trade-offs |
|-------------|--------------|---------------|-------------------|------------|
| 🚀 microAgents | **< 1MB**    | • Simple Agent & Tool Model<br>• XML-based Function Calls<br>• Universal LLM Support | **✨ Enables tool calling for ANY LLM**<br>**✨ Works with any OpenAI-compatible API**<br>**✨ Most lightweight solution** | • Bring your own tools |
| LangChain    | 166MB+       | • Complex Agent & Chain Model<br>• JSON-based Function Calls | Rich ecosystem & tooling | • Heavy footprint<br>• Complex setup |
| CrewAI       | 173MB+       | • Role-based Agents<br>• Built-in Collaboration | Agent collaboration patterns | • Complex hierarchies<br>• Heavy dependencies |
| LangGraph    | 51MB+        | • DAG-based Flows<br>• Graph Orchestration | Complex workflow support | • Steep learning curve<br>• Complex configuration |
| AutoGen      | 26MB+        | • Modular Agents<br>• Flexible Architecture | Extensible design | • Limited built-in tools |

### Why microAgents Stands Out

✅ **Ultra Lightweight**
- **Just 2K lines** of code vs 405K+ in alternatives
- **Under 1MB** vs 26MB-173MB for others
- Only two dependencies: `requests` & `urllib3`

✅ **Universal Compatibility**
- Works with **ANY** OpenAI-compatible API
- Enables tool calling even for LLMs without native support
- No vendor lock-in

✅ **Developer Experience**
- Intuitive XML-based tool calls
- Simple integration without wrapper classes
- Clean, minimalist API design

## Installation

You can install microAgents directly from PyPI:

```bash
pip install microAgents
```

Or install from source for development:

```bash
git clone https://github.com/prabhjots664/MicroAgents.git
cd MicroAgents
pip install -e .
```

## Quick Start

Here's a complete example showing how to create a multi-agent math system:

```python
from microAgents.llm import LLM
from microAgents.core import MicroAgent, Tool, MessageStore

# Initialize LLM with your API
llm = LLM(
    base_url="https://api.hyperbolic.xyz/v1",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJrYW1hbHNpbmdoZ2FsbGFAZ21haWwuY29tIiwiaWF0IjoxNzM1MjI2ODIzfQ.1wZmIzTZUWLzr-uP7Qtib_kkXNZmH_yQtSn1lP9S2z0",
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_tokens=4000,
    temperature=0.8,
    top_p=0.9
)

# Define tools for basic math operations
def add_numbers(a: float, b: float) -> float:
    return a + b

def multiply_numbers(a: float, b: float) -> float:
    return a * b

# Create specialized agents
math_agent = MicroAgent(
    llm=llm,
    prompt="You are a math assistant. Handle basic arithmetic operations.",
    toolsList=[
        Tool("add", "Add two numbers", add_numbers),
        Tool("multiply", "Multiply two numbers", multiply_numbers)
    ]
)

# Create message store for conversation history
message_store = MessageStore()

# Use the agent
response = math_agent.execute_agent(
    "First add 3 and 5, then multiply the result by 2", 
    message_store
)
print(response)
```

## Multi-Agent Orchestration Example

Here's an example of creating multiple specialized agents and orchestrating them:

```python
from microAgents.llm import LLM
from microAgents.core import MicroAgent, Tool, MessageStore

# Initialize LLM
llm = LLM(
    base_url="https://api.hyperbolic.xyz/v1",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJrYW1hbHNpbmdoZ2FsbGFAZ21haWwuY29tIiwiaWF0IjoxNzM1MjI2ODIzfQ.1wZmIzTZUWLzr-uP7Qtib_kkXNZmH_yQtSn1lP9S2z0",
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_tokens=4000,
    temperature=0.8,
    top_p=0.9
)

# Define tools for different agents
def add_numbers(a: float, b: float) -> float:
    return a + b

def multiply_numbers(a: float, b: float) -> float:
    return a * b

def factorial(n: int) -> int:
    if n == 0:
        return 1
    return n * factorial(n - 1)

# Create specialized agents
simple_math_agent = MicroAgent(
    llm=llm,
    prompt="You are a simple math assistant. Handle basic arithmetic operations.",
    toolsList=[
        Tool("add", "Add two numbers", add_numbers),
        Tool("multiply", "Multiply two numbers", multiply_numbers)
    ]
)

advanced_math_agent = MicroAgent(
    llm=llm,
    prompt="You are an advanced math assistant. Handle complex math operations.",
    toolsList=[
        Tool("factorial", "Calculate factorial", factorial)
    ]
)

# Create an orchestrator agent
class Orchestrator(MicroAgent):
    def __init__(self):
        super().__init__(
            llm=llm,
            prompt="""You are a math query analyzer. For each query:
1. If it contains basic arithmetic, output exactly: SIMPLE_MATHS NEEDED
2. If it contains advanced math, output exactly: ADVANCED_MATHS NEEDED
3. If unsure, output exactly: UNKNOWN_MATH_TYPE""",
            toolsList=[]
        )
        self.simple_math_agent = simple_math_agent
        self.advanced_math_agent = advanced_math_agent

    def execute_agent(self, query: str, message_store: MessageStore) -> str:
        # Get initial analysis from orchestrator
        analysis = super().execute_agent(query, message_store)
        
        if "SIMPLE_MATHS NEEDED" in analysis:
            result = self.simple_math_agent.execute_agent(query, message_store)
            return f"Simple Math Agent: {result}"
        elif "ADVANCED_MATHS NEEDED" in analysis:
            result = self.advanced_math_agent.execute_agent(query, message_store)
            return f"Advanced Math Agent: {result}"
        else:
            return "Unable to determine the appropriate agent for this query."

# Use the orchestrated system
message_store = MessageStore()
orchestrator = Orchestrator()

# Example queries
queries = [
    "What is 15 plus 27?",  # Will use simple_math_agent
    "Calculate 5 factorial",  # Will use advanced_math_agent
    "First add 3 and 5, then multiply the result by 2"  # Will use simple_math_agent
]

for query in queries:
    response = orchestrator.execute_agent(query, message_store)
    print(f"Query: {query}")
    print(f"Response: {response}\n")
```

This example demonstrates:
- Creating multiple specialized agents with different tools
- Building an orchestrator agent to route queries
- Using a message store to maintain conversation history
- Coordinating multiple agents to handle different types of tasks

## Examples

- `math_demo.py`: Basic math operations using tool calls

## License

MIT License