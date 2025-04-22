# microAgents Framework

A lightweight framework for building AI agents with XML-style tool calls.

## Features

- XML-style tool calling format for clear parameter structure
- Support for OpenAI-compatible LLM APIs
- Simple message history management
- Type-safe tool parameter handling
- Easy-to-use agent configuration

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from microAgents.core import MicroAgent, Tool, MessageStore
from microAgents.llm import LLM

# Define a tool
def add_numbers(a: float, b: float) -> float:
    return a + b

# Create an agent
llm = LLM(base_url="https://api.openai.com/v1")
agent = MicroAgent(
    llm=llm,
    prompt="You are a math assistant",
    toolsList=[Tool("add", "Add two numbers", add_numbers)]
)

# Use the agent
message_store = MessageStore()
response = agent.execute_agent("What is 5 plus 3?", message_store)
print(response)
```

## Tool Call Format

Tool use is formatted using XML-style tags. The tool name is enclosed in opening and closing tags, and each parameter is similarly enclosed within its own set of tags:

```xml
<TOOL_CALLS_NEEDED>
<tool_name>
<param1>value1</param1>
<param2>value2</param2>
</tool_name>
</TOOL_CALLS_NEEDED>
```

Multiple tool calls can be included:

```xml
<TOOL_CALLS_NEEDED>
<add>
<a>5</a>
<b>3</b>
</add>
<multiply>
<a>4</a>
<b>2</b>
</multiply>
</TOOL_CALLS_NEEDED>
```

## Examples

Check out the `examples/` directory for complete examples:
- `math_demo.py`: Basic math operations using tool calls

## License

MIT License