"""Module for interacting with OpenAI-like LLMs."""

__all__ = ['LLM']

import os
import logging
import requests
import urllib3
from typing import Optional, Dict, Any, List, Union

# Disable InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class LLM:
    """Base class for interacting with OpenAI-like LLM APIs."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, 
                 model: str = "gpt-3.5-turbo", max_tokens: int = 1000,
                 temperature: float = 0.7, top_p: float = 1.0):
        """
        Initialize LLM client.
        
        Args:
            base_url: Base URL of the LLM API
            api_key: API key for authentication (optional)
            model: Model name to use (default: "gpt-3.5-turbo")
            max_tokens: Maximum number of tokens to generate (default: 1000)
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 1.0)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv('LLM_API_KEY')
        self.model = model
        self.default_params = {
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p
        }
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        if self.api_key:
            self.session.headers['Authorization'] = f'Bearer {self.api_key}'
            
        self.logger = logging.getLogger(__name__)

    def _request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the LLM API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.post(url, json=payload, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
            
    def completions(self, prompt: str, max_tokens: Optional[int] = None,
                  temperature: Optional[float] = None, top_p: Optional[float] = None,
                  **kwargs) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            max_tokens: Override default max tokens
            temperature: Override default temperature
            top_p: Override default top_p
            **kwargs: Additional parameters for the completion
            
        Returns:
            Generated text
        """
        payload = {
            "prompt": prompt,
            **{k: v for k, v in self.default_params.items() if k != 'model'},
            **kwargs
        }
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens
        if temperature is not None:
            payload['temperature'] = temperature
        if top_p is not None:
            payload['top_p'] = top_p
            
        response = self._request('/completions', payload)
        return response['choices'][0]['text']
        
    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None,
            temperature: Optional[float] = None, top_p: Optional[float] = None,
            **kwargs) -> str:
        """
        Generate chat response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Override default max tokens
            temperature: Override default temperature
            top_p: Override default top_p
            **kwargs: Additional parameters for the chat
            
        Returns:
            Generated response
        """
        payload = {
            "messages": messages,
            **self.default_params,
            **kwargs
        }
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens
        if temperature is not None:
            payload['temperature'] = temperature
        if top_p is not None:
            payload['top_p'] = top_p
            
        response = self._request('/chat/completions', payload)
        return response['choices'][0]['message']['content']
        
    def embeddings(self, input: Union[str, List[str]], model: Optional[str] = None,
                 **kwargs) -> List[float]:
        """
        Generate embeddings for input text.
        
        Args:
            input: Input text or list of texts
            model: Override default model for embeddings
            **kwargs: Additional parameters for embeddings
            
        Returns:
            List of embeddings
        """
        payload = {
            "input": input,
            "model": model or self.model,
            **kwargs
        }
        response = self._request('/embeddings', payload)
        return response['data'][0]['embedding']