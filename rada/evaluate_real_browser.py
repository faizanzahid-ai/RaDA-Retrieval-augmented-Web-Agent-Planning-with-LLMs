"""
Real RaDA Evaluation with Browser Automation

Full evaluation pipeline with:
- Real browser execution using Playwright
- DOM parsing with XPath grounding
- Retrieval-based planning (RaDA)
- Mind2Web + CompWoB benchmarks
- Paper-level metrics evaluation
"""

import asyncio
import json
import os
import time
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

from rada.core.real_rada_agent import RealRaDAAgent
from rada.datasets.mind2web_dataset import load_dataset, DatasetTask
from rada.evaluation.mind2web_evaluator import Mind2WebEvaluator, create_ground_truth_for_task

console = Console()


class RealRaDAEvaluator:
    """
    Real RaDA evaluator with browser automation
    
    Executes tasks in real browser and evaluates with paper-level metrics
    """
    
    def __init__(
        self,
        llm_model: str = "llama-3.1-8b-instant",
        headless: bool = True,
        output_dir: str = "./evaluation_results"
    ):
        """
        Initialize evaluator
        
        Args:
            llm_model: LLM model
            headless: Browser headless mode
            output_dir: Output directory
        """
        self.llm_model = llm_model
        self.headless = headless
        self.output_dir = output_dir
        
        self.evaluator = Mind2WebEvaluator(output_dir=output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("./screenshots", exist_ok=True)
        
        logger.info("RealRaDAEvaluator initialized")
    
    async def evaluate_task_with_browser(
        self,
        task: DatasetTask,
        agent: RealRaDAAgent
    ) -> Dict[str, Any]:
        """
        Evaluate task with real browser execution
        
        Args:
            task: Dataset task
            agent: RaDA agent
            
        Returns:
            Task result with metrics
        """
        start_time = time.time()
        
        try:
            # Execute task in real browser
            result = await agent.execute_task(
                task=task.task_description,
                start_url=task.start_url,
                save_screenshots=False
            )
            
            # Add task metadata
            result['task_id'] = task.task_id
            result['domain'] = task.domain
            result['difficulty'] = task.difficulty
            
            return result
            
        except Exception as e:
            logger.error(f"Task evaluation failed: {e}")
            return {
                'task_id': task.task_id,
                'task_description': task.task_description,
                'domain': task.domain,
                'difficulty': task.difficulty,
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'actions': []
            }
    
    async def evaluate_dataset(
        self,
        dataset_name: str = "mind2web",
        max_tasks: int = None
    ) -> Dict[str, Any]:
        """
        Evaluate on complete dataset
        
        Args:
            dataset_name: Dataset name
            max_tasks: Maximum tasks to evaluate
            
        Returns:
            Evaluation results
        """
        console.print(Panel(
            f"[bold blue]Evaluating {dataset_name.upper()} with Real Browser[/bold blue]",
            border_style="blue"
        ))
        
        # Load dataset
        tasks = load_dataset(dataset_name)
        if max_tasks:
            tasks = tasks[:max_tasks]
        
        console.print(f"[cyan]Tasks to evaluate: {len(tasks)}[/cyan]")
        console.print(f"[cyan]Model: {self.llm_model}[/cyan]")
        console.print(f"[cyan]Browser: {'Headless' if self.headless else 'Visible'}[/cyan]\n")
        
        # Initialize agent
        agent = RealRaDAAgent(
            llm_model=self.llm_model,
            headless=self.headless,
            max_actions_per_subtask=15,
            max_subtasks=7
        )
        
        # Evaluate each task
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task_progress = progress.add_task(
                "[cyan]Evaluating...", total=len(tasks)
            )
            
            for i, task in enumerate(tasks, 1):
                progress.update(
                    task_progress,
                    description=f"Task {i}/{len(tasks)}: {task.task_description[:60]}..."
                )
                
                result = await self.evaluate_task_with_browser(task, agent)
                results.append(result)
                
                progress.advance(task_progress)
                
                # Log progress
                status = "✓" if result.get('success') else "✗"
                console.print(f"  [{status}] Task {i}: {result.get('success', False)}")
        
        # Generate ground truths
        ground_truths = [create_ground_truth_for_task(task) for task in tasks]
        
        # Evaluate with Mind2Web metrics
        console.print("\n[yellow]Calculating paper-level metrics...[/yellow]\n")
        metrics = self.evaluator.evaluate_dataset(
            results=results,
            ground_truths=ground_truths,
            dataset_name=dataset_name
        )
        
        # Print report
        self.evaluator.print_mind2web_report(metrics)
        
        # Save detailed results
        self._save_detailed_results(results, metrics, dataset_name)
        
        return metrics
    
    def _save_detailed_results(
        self,
        results: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        dataset_name: str
    ):
        """
        Save detailed evaluation results
        
        Args:
            results: Task results
            metrics: Evaluation metrics
            dataset_name: Dataset name
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_detailed_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        detailed = {
            'metadata': {
                'dataset': dataset_name,
                'model': self.llm_model,
                'timestamp': timestamp,
                'total_tasks': len(results)
            },
            'metrics': metrics,
            'results': results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed, f, indent=2)
        
        logger.info(f"Detailed results saved to {filepath}")


async def main():
    """Main evaluation function"""
    console.print("="*80)
    console.print(Panel(
        "[bold green]REAL RaDA EVALUATION SYSTEM[/bold green]\n"
        "🔥 Real browser automation (Playwright)\n"
        "🔥 DOM parsing + XPath grounding\n"
        "🔥 Retrieval-based planning (RaDA)\n"
        "🔥 Full Mind2Web evaluator (paper metrics)\n"
        "🔥 Benchmarks: Mind2Web + CompWoB",
        border_style="green"
    ))
    console.print("="*80 + "\n")
    
    # Initialize evaluator
    evaluator = RealRaDAEvaluator(
        llm_model="llama-3.1-8b-instant",
        headless=False,  # Set to False to see browser
        output_dir="./evaluation_results"
    )
    # Evaluate on Mind2Web
    console.print("\n[bold yellow]Phase 1: Mind2Web Evaluation[/bold yellow]\n")
    mind2web_metrics = await evaluator.evaluate_dataset(
        dataset_name="mind2web",
        max_tasks=5  # Start with 5 tasks, set to None for all 10
    )
    
    # Evaluate on CompWoB
    console.print("\n[bold yellow]Phase 2: CompWoB Evaluation[/bold yellow]\n")
    compwob_metrics = await evaluator.evaluate_dataset(
        dataset_name="compwob",
        max_tasks=3  # All CompWoB tasks
    )
    
    # Final Summary
    console.print("\n" + "="*80)
    console.print(Panel(
        "[bold green]✓ EVALUATION COMPLETE[/bold green]\n\n"
        f"Mind2Web Results:\n"
        f"  Task Success Rate: {mind2web_metrics['task_success_rate']:.2%}\n"
        f"  Step Success Rate: {mind2web_metrics['step_success_rate']:.2%}\n"
        f"  Action F1: {mind2web_metrics['action_f1']:.4f}\n\n"
        f"CompWoB Results:\n"
        f"  Task Success Rate: {compwob_metrics['task_success_rate']:.2%}\n"
        f"  Step Success Rate: {compwob_metrics['step_success_rate']:.2%}\n"
        f"  Action F1: {compwob_metrics['action_f1']:.4f}\n\n"
        f"Results saved to: ./evaluation_results/",
        border_style="green"
    ))
    console.print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
