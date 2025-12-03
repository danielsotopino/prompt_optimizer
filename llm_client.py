"""
OpenRouter LLM Client
Simplified client for OpenRouter API only
"""
from typing import Optional
from openai import OpenAI
import os

class LLMClient:
    """
    OpenRouter LLM client
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize OpenRouter LLM client
        
        Args:
            api_key: OpenRouter API key (if None, reads from OPENROUTER_API_KEY env var)
            base_url: Custom base URL (defaults to OpenRouter API)
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY not found. Set it as environment variable or pass api_key parameter."
                )
        
        # Set base URL
        if base_url is None:
            base_url = "https://openrouter.ai/api/v1"
        
        # Initialize OpenAI client (compatible with OpenRouter API)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Store info
        self.api_key = api_key
        self.base_url = base_url
    
    def chat_completions_create(
        self,
        model: str,
        messages: list,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Create chat completion using OpenRouter API
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3.5-sonnet")
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            max_completion_tokens: Alternative param name (normalized to max_tokens)
            **kwargs: Additional parameters
        
        Returns:
            Chat completion response
        """
        # Normalize max_tokens parameter
        if max_completion_tokens is not None and max_tokens is None:
            max_tokens = max_completion_tokens
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # Add OpenRouter-specific headers
        extra_headers = kwargs.get("extra_headers", {})
        if "HTTP-Referer" not in extra_headers:
            extra_headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERER", "https://github.com/prompt-optimizer")
        if "X-Title" not in extra_headers:
            extra_headers["X-Title"] = "Prompt Optimizer"
        params["extra_headers"] = extra_headers
        
        return self.client.chat.completions.create(**params)
    
    def __repr__(self):
        return f"LLMClient(base_url={self.base_url})"


def create_llm_client(api_key: Optional[str] = None) -> LLMClient:
    """
    Factory function to create OpenRouter LLM client
    
    Args:
        api_key: OpenRouter API key (if None, reads from OPENROUTER_API_KEY env var)
    
    Returns:
        Configured LLMClient instance
    """
    return LLMClient(api_key=api_key)
