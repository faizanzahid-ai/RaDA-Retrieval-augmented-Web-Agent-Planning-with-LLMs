"""
Example: Using different models with RaDA
"""

import asyncio
from rada.main import RADA
from rada.config import Config

async def example_with_config():
    """Example using config to auto-select model"""
    print("=" * 60)
    print("Example 1: Auto-select model based on API keys")
    print("=" * 60)
    
    # Will automatically select Groq if GROQ_API_KEY is set
    # Otherwise falls back to OpenAI if OPENAI_API_KEY is set
    agent = RADA()
    
    result = await agent.execute_task(
        task="Navigate to example.com and verify page loads",
        start_url="https://example.com"
    )
    print(f"Result: {result}")

async def example_with_groq():
    """Example explicitly using Groq"""
    print("\n" + "=" * 60)
    print("Example 2: Explicitly use Groq model")
    print("=" * 60)
    
    # Use a specific Groq model
    agent = RADA(
        llm_model="llama-3.3-70b-versatile"  # Specific Groq model
    )
    
    result = await agent.execute_task(
        task="Navigate to example.com and verify page loads",
        start_url="https://example.com"
    )
    print(f"Result: {result}")

async def example_with_openai():
    """Example explicitly using OpenAI"""
    print("\n" + "=" * 60)
    print("Example 3: Explicitly use a different model")
    print("=" * 60)
    
    # Use a different model
    agent = RADA(
        llm_model="mixtral-8x7b-32768"  # Another Groq model
    )
    
    result = await agent.execute_task(
        task="Navigate to example.com and verify page loads",
        start_url="https://example.com"
    )
    print(f"Result: {result}")

async def example_list_models():
    """Example listing available models"""
    print("\n" + "=" * 60)
    print("Available Models")
    print("=" * 60)
    
    print("\nGroq Models:")
    for model in Config.GROQ_MODELS:
        print(f"  - {model}")
    
    print(f"\nCurrently set: {Config.DEFAULT_LLM_MODEL}")

if __name__ == "__main__":
    # List available models
    asyncio.run(example_list_models())
    
    # Run example with config (auto-select)
    # asyncio.run(example_with_config())
    
    # Run example with specific model
    # asyncio.run(example_with_groq())
    # asyncio.run(example_with_openai())
