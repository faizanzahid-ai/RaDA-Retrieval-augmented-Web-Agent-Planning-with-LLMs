"""
OpenRouter API client for LLM inference (OpenAI-compatible)
Supports multiple models through OpenRouter's unified API
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai package not installed. Install with: pip install openai")


class OpenRouterClient:
    """OpenRouter API client for LLM inference"""
    
    def __init__(
        self,
        model: str = "meta-llama/llama-3.1-8b-instruct",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize OpenRouter client
        
        Args:
            model: OpenRouter model to use (format: provider/model)
            api_key: OpenRouter API key (optional, uses env var if not provided)
            base_url: OpenRouter API base URL
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
        
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Please set it or pass api_key parameter."
            )
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        logger.info(f"Initialized OpenRouter client with model: {model}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3
    ) -> str:
        """
        Create a chat completion with retry logic
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retries
            
        Returns:
            Generated text response
        """
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.95
                )
                
                content = response.choices[0].message.content or ""
                return content.strip()
                
            except Exception as e:
                error_str = str(e)
                # Retry on rate limit errors (429)
                if '429' in error_str or 'rate_limit' in error_str.lower():
                    if attempt < max_retries:
                        wait_time = 10.0 * (attempt + 1)
                        logger.warning(f"Rate limit hit, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                logger.error(f"OpenRouter API error: {e}")
                raise
        
        raise RuntimeError(f"Failed after {max_retries} retries")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of popular OpenRouter models
        
        Returns:
            List of model names
        """
        return [
            "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.3-70b-instruct",
            "google/gemma-2-9b-it",
            "google/gemma-2-27b-it",
            "mistralai/mistral-7b-instruct",
            "mistralai/mixtral-8x7b-instruct",
            "deepseek/deepseek-chat",
            "qwen/qwen-2.5-72b-instruct",
            "anthropic/claude-3.5-sonnet"
        ]
