"""System prompt template for microAgents tool integration."""

def get_postfix_system_prompt(tools_schema: dict) -> str:
    """Generate a system prompt postfix describing available tools.
    
    Args:
        tools_schema: Dictionary containing tool information from get_tools_schema()
        
    Returns:
        str: System prompt postfix describing available tools and usage
    """
    tools_description = []
    for name, info in tools_schema.items():
        param_descriptions = []
        for param, param_info in info['parameters'].items():
            param_type = param_info['type'].__name__
            param_descriptions.append(f"- {param}: {param_type}")
        tools_description.append(
            f"- **{name}**: {info['description']}\n"
            f"      Parameters:\n      " + "\n      ".join(param_descriptions)
        )
    tools_description = "\n\n".join(tools_description)
    
    return f"""
You have access to the following tools that you can use by enclosing them in 
<TOOL_CALLS_NEEDED> tags. Each tool and its parameters should be enclosed in XML-style tags.

Available tools:
{tools_description}

Example usage:
<TOOL_CALLS_NEEDED>
<tool_name>
<param1>value1</param1>
<param2>value2</param2>
</tool_name>
</TOOL_CALLS_NEEDED>

Multiple tool calls example:
<TOOL_CALLS_NEEDED>
<tool1>
<param1>value1</param1>
</tool1>
<tool2>
<param1>value1</param1>
<param2>value2</param2>
</tool2>
</TOOL_CALLS_NEEDED>

Important notes:
- If you are not provided any tool, don't respond with <TOOL_CALLS_NEEDED> tags
- Only use provided tools when absolutely necessary
- All parameters must be provided in their own XML tags
- Tool calls must be enclosed in <TOOL_CALLS_NEEDED> tags
- Parameters should use their exact parameter names as tag names
"""