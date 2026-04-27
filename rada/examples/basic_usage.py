"""
Example usage of RaDA agent
"""

import asyncio
import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


async def example_basic_usage():
    """Basic example of using RaDA agent"""
    from rada.core.agent import RadaAgent
    
    # Initialize agent
    agent = RadaAgent(
        llm_model="llama-3.1-8b-instant",  # Groq model
        embedding_model="all-MiniLM-L6-v2",
        api_key=os.getenv("GROQ_API_KEY"),
        headless=False  # Set to True for headless mode
    )
    
    # Execute a task
    task = "Search for Python programming tutorials"
    start_url = "https://www.google.com"
    
    logger.info(f"Executing task: {task}")
    
    result = await agent.execute_task(
        task=task,
        start_url=start_url
    )
    
    logger.info(f"Task result: {result}")
    return result


async def example_with_exemplars():
    """Example demonstrating exemplar storage and retrieval"""
    from rada.core.agent import RadaAgent
    
    agent = RadaAgent(
        llm_model="llama-3.1-8b-instant",  # Groq model
        api_key=os.getenv("GROQ_API_KEY"),
        headless=True
    )
    
    # Manually add an exemplar
    await agent.add_exemplar_manually(
        task="Search for information",
        subtask="Navigate to search engine",
        actions=[
            {"action": "NAVIGATE", "value": "https://www.google.com"},
            {"action": "WAIT", "value": "1000"}
        ],
        outcome="Successfully navigated to Google",
        context="Search task initialization"
    )
    
    logger.info("Added exemplar manually")


async def main():
    """Run examples"""
    logger.info("Starting RaDA examples")
    
    # Run basic example
    try:
        await example_basic_usage()
    except Exception as e:
        logger.error(f"Basic example failed: {e}")
    
    # Run exemplar example
    try:
        await example_with_exemplars()
    except Exception as e:
        logger.error(f"Exemplar example failed: {e}")
    
    logger.info("Examples completed")


if __name__ == "__main__":
    asyncio.run(main())
