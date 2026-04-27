"""
Configuration for RaDA models and settings
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

class Config:
    """Configuration for RaDA agent"""
    
    # Model settings
    DEFAULT_LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"  # OpenRouter model (default)
    LLM_PROVIDER = "openrouter"  # Options: "openrouter", "nvidia", "groq"
    
    # OpenRouter models
    OPENROUTER_MODELS = [
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "google/gemma-2-9b-it",
        "mistralai/mixtral-8x7b-instruct"
    ]
    
    # NVIDIA models
    NVIDIA_MODELS = [
        "deepseek-ai/deepseek-v4-flash",
        "deepseek-ai/deepseek-r1",
        "meta/llama-3.1-70b-instruct"
    ]
    
    # Groq models
    GROQ_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    
    # Check which API keys are available
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    
    @classmethod
    def get_available_model(cls):
        """Get the best available model based on API keys"""
        # Priority: OpenRouter > NVIDIA > Groq
        if cls.OPENROUTER_API_KEY:
            cls.LLM_PROVIDER = "openrouter"
            return cls.DEFAULT_LLM_MODEL
        elif cls.NVIDIA_API_KEY:
            cls.LLM_PROVIDER = "nvidia"
            return "deepseek-ai/deepseek-v4-flash"
        elif cls.GROQ_API_KEY:
            cls.LLM_PROVIDER = "groq"
            return "llama-3.1-8b-instant"
        else:
            raise ValueError(
                "No API keys found. Please set OPENROUTER_API_KEY, NVIDIA_API_KEY, or GROQ_API_KEY."
            )
    
    @classmethod
    def set_model(cls, model_name: str, provider: str = None):
        """Set the model to use"""
        cls.DEFAULT_LLM_MODEL = model_name
        if provider:
            cls.LLM_PROVIDER = provider
