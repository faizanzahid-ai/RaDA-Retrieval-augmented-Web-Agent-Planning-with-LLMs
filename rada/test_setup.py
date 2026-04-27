"""
Simple test to verify Groq API integration
"""

import asyncio
import os
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console

load_dotenv()

console = Console()


async def test_groq_api():
    """Test LLM API integration (NVIDIA or Groq)"""
    console.print("[bold blue]Testing LLM API Integration...[/bold blue]\n")
    
    try:
        from rada.config import Config
        from rada.utils.openrouter_client import OpenRouterClient
        from rada.utils.nvidia_client import NVIDIAClient
        from rada.utils.groq_client import GroqClient
        
        # Get model and provider from config
        llm_model = Config.get_available_model()
        provider = Config.LLM_PROVIDER
        
        # Initialize client based on provider
        if provider.lower() == "openrouter":
            client = OpenRouterClient(model=llm_model)
        elif provider.lower() == "nvidia":
            client = NVIDIAClient(model=llm_model)
        else:
            client = GroqClient(model=llm_model)
        
        console.print(f"✅ {provider.upper()} client initialized with {llm_model}\n")
        
        # Test chat completion
        console.print("[cyan]Testing chat completion...[/cyan]")
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in one sentence!"}
            ],
            temperature=0.5,
            max_tokens=100
        )
        
        console.print(f"[green]✅ Response received:[/green] {response}\n")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return False


async def test_rada_components():
    """Test RaDA components"""
    console.print("[bold blue]Testing RaDA Components...[/bold blue]\n")
    
    try:
        from rada.config import Config
        from rada.planner.rad import RaDPlanner
        from rada.executor.raa import RaAExecutor
        from rada.utils.openrouter_client import OpenRouterClient
        from rada.utils.nvidia_client import NVIDIAClient
        from rada.utils.groq_client import GroqClient
        
        # Get model and provider from config
        llm_model = Config.get_available_model()
        provider = Config.LLM_PROVIDER
        
        # Initialize client
        if provider.lower() == "openrouter":
            client = OpenRouterClient(model=llm_model)
        elif provider.lower() == "nvidia":
            client = NVIDIAClient(model=llm_model)
        else:
            client = GroqClient(model=llm_model)
        
        # Test RaD
        console.print("[cyan]Testing RaD (Task Decomposition)...[/cyan]")
        rad = RaDPlanner(
            llm_client=client,
            benchmark="compwob"
        )
        
        subtasks = rad.decompose(
            task="Search for Python tutorials on Google",
            url="https://www.google.com",
            page_title="Google",
            page_summary="Google search homepage"
        )
        
        console.print(f"[green]✅ RaD working! Decomposed into {len(subtasks)} subtasks:[/green]")
        for i, subtask in enumerate(subtasks, 1):
            console.print(f"   {i}. {subtask}")
        console.print()
        
        # Test RaA
        console.print("[cyan]Testing RaA (Action Generation)...[/cyan]")
        raa = RaAExecutor(
            llm_client=client,
            benchmark="compwob"
        )
        
        action = raa.generate_action(
            task="Search for Python tutorials",
            subtask="Navigate to Google search page",
            remaining_subtasks=["Navigate to Google search page", "Enter search query", "Click search button"],
            trajectory="Current URL: https://www.google.com\nPage Title: Google\nElements: - [search_box] <input> text: '' type: 'text'",
            action_history=[],
            retrieved_exemplars="No exemplars available."
        )
        
        console.print(f"[green]✅ RaA working! Generated action:[/green]")
        console.print(f"   Action: {action.get('action')}")
        console.print(f"   Reasoning: {action.get('reasoning')}")
        console.print()
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def test_dataset():
    """Test dataset loading"""
    console.print("[bold blue]Testing Dataset Loading...[/bold blue]\n")
    
    try:
        from rada.datasets.mind2web_dataset import Mind2WebDataset, CompWoBDataset
        
        # Test Mind2Web
        console.print("[cyan]Loading Mind2Web dataset...[/cyan]")
        mind2web = Mind2WebDataset()
        tasks = mind2web.get_all_tasks()
        stats = mind2web.get_statistics()
        
        console.print(f"[green]✅ Mind2Web loaded: {stats['total_tasks']} tasks[/green]")
        console.print(f"   Difficulty distribution: {stats['difficulty_distribution']}")
        console.print(f"   Domains: {stats['domains'][:5]}...")
        console.print()
        
        # Test CompWoB
        console.print("[cyan]Loading CompWoB dataset...[/cyan]")
        compwob = CompWoBDataset()
        compwob_tasks = compwob.get_all_tasks()
        
        console.print(f"[green]✅ CompWoB loaded: {len(compwob_tasks)} tasks[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    console.print("="*80)
    console.print("[bold green]RaDA Implementation Test Suite[/bold green]")
    console.print("="*80 + "\n")
    
    # Test 1: Groq API
    groq_ok = await test_groq_api()
    
    # Test 2: RaDA Components
    rada_ok = await test_rada_components()
    
    # Test 3: Dataset
    dataset_ok = await test_dataset()
    
    # Summary
    console.print("\n" + "="*80)
    console.print("[bold]Test Summary:[/bold]")
    console.print(f"  LLM API: {'✅ PASS' if groq_ok else '❌ FAIL'}")
    console.print(f"  RaDA Components: {'✅ PASS' if rada_ok else '❌ FAIL'}")
    console.print(f"  Dataset: {'✅ PASS' if dataset_ok else '❌ FAIL'}")
    console.print("="*80 + "\n")
    
    if groq_ok and rada_ok and dataset_ok:
        console.print("[bold green]🎉 All tests passed! Ready to run evaluation.[/bold green]\n")
        console.print("Next step: Run 'python evaluate_with_dataset.py'\n")
    else:
        console.print("[bold red]⚠️  Some tests failed. Check the errors above.[/bold red]\n")


if __name__ == "__main__":
    asyncio.run(main())
