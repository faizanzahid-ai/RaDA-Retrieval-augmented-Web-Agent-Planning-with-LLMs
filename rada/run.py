"""
RaDA Runner

Run complete evaluation on Mind2Web and CompWoB benchmarks
"""

import asyncio
import sys
import warnings
from rich.console import Console
from rich.panel import Panel

from rada.benchmarks.run_mind2web import run_mind2web
from rada.benchmarks.run_compwob import run_compwob
from rada.utils import setup_logging, ensure_dirs, format_metrics
from rada.evaluation.paper_report import RaDAPaperReport

# Suppress warnings
warnings.filterwarnings("ignore")

console = Console()


async def main():
    """Run benchmarks"""
    console.print(Panel(
        "[bold green]RaDA: Retrieval-augmented Web Agent Planning[/bold green]\n"
        "Running Mind2Web + CompWoB Benchmarks",
        border_style="green"
    ))
    
    # Setup
    setup_logging()
    ensure_dirs()
    
    try:
        # Run Mind2Web
        console.print("\n[bold yellow]Phase 1: Mind2Web Benchmark[/bold yellow]\n")
        mind2web_metrics, mind2web_results = await run_mind2web(max_tasks=1, return_results=True)
        
        console.print("\n[bold green]Mind2Web Results:[/bold green]")
        console.print(format_metrics(mind2web_metrics))
        
        # Run CompWoB
        console.print("\n[bold yellow]Phase 2: CompWoB Benchmark[/bold yellow]\n")
        compwob_metrics, compwob_results = await run_compwob(max_tasks=1, return_results=True)
        
        console.print("\n[bold green]CompWoB Results:[/bold green]")
        console.print(format_metrics(compwob_metrics))
        
        # Generate paper-style report
        console.print("\n[bold yellow]Phase 3: Generating Paper-Style Report[/bold yellow]\n")
        from rada.config import Config
        report_generator = RaDAPaperReport()
        report_file = report_generator.generate_comprehensive_report(
            mind2web_results=mind2web_results,
            compwob_results=compwob_results,
            model_name=Config.DEFAULT_LLM_MODEL,
            provider=Config.LLM_PROVIDER
        )
        
        # Summary
        console.print("\n" + "="*60)
        console.print(Panel(
            "[bold green]✓ Evaluation Complete![/bold green]\n\n"
            f"Mind2Web Success: {mind2web_metrics.get('task_success_rate', 0):.2%}\n"
            f"CompWoB Success: {compwob_metrics.get('success_rate', 0):.2%}\n\n"
            f"Paper Report: {report_file}",
            border_style="green"
        ))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
