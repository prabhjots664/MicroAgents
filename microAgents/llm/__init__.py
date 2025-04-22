"""LLM integration components for the microAgents framework."""

from .llm import LLM
from .prompt import get_postfix_system_prompt

__all__ = ['LLM', 'get_postfix_system_prompt']