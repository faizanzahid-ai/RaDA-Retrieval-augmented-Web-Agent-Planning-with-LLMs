"""
NVIDIA API client for LLM inference (OpenAI-compatible)
Supports DeepSeek V4 Flash and other NVIDIA NIM models
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


class NVIDIAClient:
    """NVIDIA API client for LLM inference"""
    
    def __init__(
        self,
        model: str = "deepseek-ai/deepseek-v4-flash",
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1"
    ):
        """
        Initialize NVIDIA client
        
        Args:
            model: NVIDIA NIM model to use
            api_key: NVIDIA API key (optional, uses env var if not provided)
            base_url: NVIDIA API base URL
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
        
        self.model = model
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY environment variable is not set. "
                "Please set it or pass api_key parameter."
            )
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        logger.info(f"Initialized NVIDIA client with model: {model}")
    
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
                # Build extra_body for DeepSeek models with reasoning
                extra_body = {}
                if "deepseek" in self.model.lower():
                    # Disable reasoning for faster response (can enable if needed)
                    extra_body = {
                        "chat_template_kwargs": {
                            "thinking": False,
                            "reasoning_effort": "low"
                        }
                    }
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.95,
                    extra_body=extra_body if extra_body else None
                )
                
                # Extract reasoning content if available (for DeepSeek)
                content = response.choices[0].message.content or ""
                reasoning = getattr(response.choices[0].message, "reasoning", None) or \
                           getattr(response.choices[0].message, "reasoning_content", None)
                
                if reasoning:
                    logger.debug(f"Reasoning content: {reasoning[:200]}...")
                
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
                logger.error(f"NVIDIA API error: {e}")
                raise
        
        raise RuntimeError(f"Failed after {max_retries} retries")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available NVIDIA NIM models
        
        Returns:
            List of model names (hardcoded common models)
        """
        return [
            "deepseek-ai/deepseek-v4-flash",
            "deepseek-ai/deepseek-r1",
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-405b-instruct",
            "mistralai/mistral-large-2",
            "google/gemma-2-27b-it",
            "microsoft/phi-3-medium-128k-instruct"
        ]
