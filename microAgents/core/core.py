"""Core functionality for the microAgents framework."""

__all__ = ['Tool', 'MicroAgent']

import re
from typing import Callable, Dict, Any, List
import inspect
from xml.etree import ElementTree as ET
from microAgents.llm.llm import LLM
from microAgents.llm.prompt import get_postfix_system_prompt
from .message_store import BaseMessageStore

class Tool:
    """Base class for microAgents tools."""
    
    def __init__(self, description: str, func: Callable, name: str = None):
        self.name = name or func.__name__
        self.description = description
        self.func = func
        self.parameters = self._extract_parameters()
        
    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract parameter information from the function signature."""
        sig = inspect.signature(self.func)
        parameters = {}
        
        for name, param in sig.parameters.items():
            parameters[name] = {
                'type': param.annotation if param.annotation != inspect.Parameter.empty else str,
                'required': param.default == inspect.Parameter.empty
            }
            
        return parameters
        
    def execute(self, **kwargs) -> Any:
        """Execute the tool with the provided arguments."""
        return self.func(**kwargs)

class MicroAgent:
    """Main agent class managing tools, LLM interactions, and state."""
    
    def __init__(self, llm: LLM, prompt: str, toolsList: List[Tool]):
        self.tools: Dict[str, Tool] = {}
        self.llm = llm
        self.initial_prompt = prompt
        
        # Register provided tools
        for tool in toolsList:
            self.register_tool(tool)
            
    def execute_agent(self, user_input: str, message_store: BaseMessageStore) -> str:
        """Execute the agent's reasoning and tool usage."""
        # print(f"\nDEBUG: Starting execute_agent with input: {user_input}")
        
        # Prepare messages for LLM interaction
        messages = [
            {"role": "system",
             "content": self.initial_prompt + "\n" + get_postfix_system_prompt(self.get_tools_schema())}
        ]

        # Add the new user input to message store
        if user_input:
            message_store.add_message({
            'role': 'user',
            'content': user_input
        })
        
        # Add conversation history
        messages.extend(message_store.get_messages())

        # Get LLM response
        response = self.llm.chat(messages)
        # print(f"DEBUG: LLM response: {response}")

        
        # If response contains tool calls, execute them and get results
        if "<TOOL_CALLS_NEEDED>" in response:
            # print("DEBUG: Found tool calls in response")
            tool_calls = self._parse_tool_calls(response)
            # print(f"DEBUG: Parsed tool calls: {tool_calls}")
            results = []
            
            for call in tool_calls:
                try:
                    # print(f"DEBUG: Executing tool {call['name']} with params {call['params']}")
                    result = self.execute_tool(call['name'], **call['params'])
                    # print(f"DEBUG: Tool execution result: {result}")
                    results.append(f"Tool {call['name']} result: {result}")
                except Exception as e:
                    # print(f"DEBUG: Tool execution error: {str(e)}")
                    results.append(f"Tool {call['name']} error: {str(e)}")

            # Add tool call results to message store
            for result in results:
                message_store.add_message({
                    'role': 'user',
                    'content': result
                })
            
            # print("DEBUG: Making recursive call with tool results")
            # Recursively call execute_agent with the same input to get final response
            return self.execute_agent('', message_store)
        else:
            # print("DEBUG: No tool calls needed, returning response")
            # No tool calls needed, just return the response
            message_store.add_message({
                'role': 'assistant',
                'content': response
            })
            return response

    def register_tool(self, tool: Tool):
        """Register a new tool with the framework."""
        self.tools[tool.name] = tool
        
    def get_tools_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for all registered tools."""
        return {
            tool.name: {
                'description': tool.description,
                'parameters': tool.parameters
            }
            for tool in self.tools.values()
        }
        
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a registered tool by name with named arguments."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        tool = self.tools[tool_name]
        return tool.execute(**kwargs)

    def _parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse XML-style tool calls from content.
        
        Example format:
        <TOOL_CALLS_NEEDED>
        <tool_name>
        <param1>value1</param1>
        <param2>value2</param2>
        </tool_name>
        </TOOL_CALLS_NEEDED>
        """
        calls = []
        pattern = r"<TOOL_CALLS_NEEDED>(.*?)</TOOL_CALLS_NEEDED>"
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            # Create valid XML by wrapping in root tag
            xml_str = f"<root>{match}</root>"
            try:
                root = ET.fromstring(xml_str)
                # Each direct child of root is a tool call
                for tool_elem in root:
                    tool_name = tool_elem.tag
                    if tool_name not in self.tools:
                        raise ValueError(f"Unknown tool: {tool_name}")
                        
                    tool = self.tools[tool_name]
                    params = {}
                    # Each child of tool element is a parameter
                    for param_elem in tool_elem:
                        param_name = param_elem.tag
                        param_value = param_elem.text.strip()
                        
                        # Convert value to the parameter's specified type
                        if param_name in tool.parameters:
                            param_type = tool.parameters[param_name]['type']
                            try:
                                # Convert the string value to the parameter's type
                                params[param_name] = param_type(param_value)
                            except (ValueError, TypeError) as e:
                                raise ValueError(f"Failed to convert parameter '{param_name}' value '{param_value}' to type {param_type.__name__}: {str(e)}")
                        else:
                            raise ValueError(f"Unknown parameter '{param_name}' for tool '{tool_name}'")
                            
                    calls.append({
                        'name': tool_name,
                        'params': params
                    })
            except ET.ParseError as e:
                raise ValueError(f"Invalid XML format in tool calls: {str(e)}")
                
        return calls