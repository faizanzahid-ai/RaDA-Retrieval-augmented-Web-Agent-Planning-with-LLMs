"""
Groq API client for LLM inference (OpenAI-compatible)
"""

import os
import time
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed. Install with: pip install groq")


class GroqClient:
    """Groq API client for fast LLM inference"""
    
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None
    ):
        """
        Initialize Groq client
        
        Args:
            model: Groq model to use
            api_key: Groq API key (optional, uses env var if not provided)
        """
        if not GROQ_AVAILABLE:
            raise ImportError(
                "groq package is required. Install with: pip install groq"
            )
        
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please set it or pass api_key parameter."
            )
        
        self.client = Groq(api_key=self.api_key)
        logger.info(f"Initialized Groq client with model: {model}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 500,
        max_retries: int = 3
    ) -> str:
        """
        Create a chat completion with retry logic for rate limits
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retries for rate limit errors
            
        Returns:
            Generated text response
        """
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                error_str = str(e)
                # Retry on rate limit errors (429)
                if '429' in error_str or 'rate_limit' in error_str.lower():
                    if attempt < max_retries:
                        # Extract wait time from error message, default 15s
                        wait_time = 15.0
                        try:
                            import re
                            match = re.search(r'try again in ([\d.]+)s', error_str)
                            if match:
                                wait_time = float(match.group(1)) + 1.0
                        except (ValueError, AttributeError):
                            pass
                        logger.warning(f"Rate limit hit, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                logger.error(f"Groq API error: {e}")
                raise
        
        # Should not reach here, but just in case
        raise RuntimeError(f"Failed after {max_retries} retries")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models
        
        Returns:
            List of model names
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
