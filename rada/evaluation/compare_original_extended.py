"""
Comparison Evaluation: Original RaDA vs Extended RaDA

Compares original RaDA implementation with extended features:
- Graph-based task decomposition
- Adaptive memory system
- RL + RaDA learning loop
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from rada.core.real_rada_agent import RealRaDAAgent
from rada.core.rl_agent import RLRaDAgent
from rada.datasets.mind2web_dataset import load_dataset


console = Console()


class ComparisonEvaluator:
    """Evaluator for comparing original vs extended RaDA"""
    
    def __init__(self, output_dir: str = "./comparison_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.original_results = []
        self.extended_results = []
    
    async def evaluate_original(
        self,
        tasks: List[Dict],
        max_tasks: int = None
    ) -> List[Dict]:
        """Evaluate original RaDA implementation"""
        console.print(Panel(
            "[bold blue]Evaluating ORIGINAL RaDA[/bold blue]",
            border_style="blue"
        ))
        
        agent = RealRaDAAgent(
            llm_model="llama-3.1-8b-instant",
            headless=True,
            max_actions_per_subtask=10
        )
        
        results = []
        if max_tasks:
            tasks = tasks[:max_tasks]
        
        for i, task in enumerate(tasks, 1):
            console.print(f"[cyan]Task {i}/{len(tasks)}: {task['task_description'][:50]}...[/cyan]")
            
            start_time = time.time()
            try:
                result = await agent.execute_task(
                    task=task['task_description'],
                    start_url=task['start_url'],
                    save_screenshots=False
                )
                execution_time = time.time() - start_time
                
                results.append({
                    'task_id': task.get('task_id', i),
                    'task': task['task_description'],
                    'success': result.get('success', False),
                    'execution_time': execution_time,
                    'num_actions': len(result.get('actions', [])),
                    'version': 'original'
                })
                
                status = "✓" if result.get('success') else "✗"
                console.print(f"  [{status}] Success: {result.get('success')}, Time: {execution_time:.2f}s")
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Task failed: {e}")
                results.append({
                    'task_id': task.get('task_id', i),
                    'task': task['task_description'],
                    'success': False,
                    'execution_time': execution_time,
                    'num_actions': 0,
                    'error': str(e),
                    'version': 'original'
                })
                console.print(f"  [✗] Failed: {str(e)[:50]}")
        
        self.original_results = results
        return results
    
    async def evaluate_extended(
        self,
        tasks: List[Dict],
        max_tasks: int = None,
        num_episodes: int = 3
    ) -> List[Dict]:
        """Evaluate extended RaDA with RL and adaptive memory"""
        console.print(Panel(
            "[bold green]Evaluating EXTENDED RaDA (RL + Adaptive Memory)[/bold green]",
            border_style="green"
        ))
        
        agent = RLRaDAgent(
            llm_model="llama-3.1-8b-instant",
            headless=True,
            exemplar_store_path="./adaptive_exemplars",
            learning_rate=0.01
        )
        
        results = []
        if max_tasks:
            tasks = tasks[:max_tasks]
        
        for episode in range(num_episodes):
            console.print(f"\n[yellow]Episode {episode + 1}/{num_episodes}[/yellow]")
            
            for i, task in enumerate(tasks, 1):
                console.print(f"[cyan]Task {i}/{len(tasks)}: {task['task_description'][:50]}...[/cyan]")
                
                start_time = time.time()
                try:
                    result = await agent.execute_task(
                        task=task['task_description'],
                        start_url=task['start_url'],
                        explore=(episode == 0)  # Explore in first episode
                    )
                    
                    results.append({
                        'task_id': task.get('task_id', i),
                        'task': task['task_description'],
                        'success': result.get('success', False),
                        'reward': result.get('reward', 0),
                        'execution_time': result.get('execution_time', 0),
                        'num_actions': result.get('num_actions', 0),
                        'episode': episode + 1,
                        'version': 'extended'
                    })
                    
                    status = "✓" if result.get('success') else "✗"
                    console.print(f"  [{status}] Success: {result.get('success')}, Reward: {result.get('reward', 0):.3f}")
                    
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    results.append({
                        'task_id': task.get('task_id', i),
                        'task': task['task_description'],
                        'success': False,
                        'reward': 0,
                        'execution_time': 0,
                        'num_actions': 0,
                        'episode': episode + 1,
                        'error': str(e),
                        'version': 'extended'
                    })
                    console.print(f"  [✗] Failed: {str(e)[:50]}")
        
        self.extended_results = results
        return results
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute evaluation metrics"""
        if not results:
            return {}
        
        successful = [r for r in results if r.get('success', False)]
        success_rate = len(successful) / len(results)
        
        avg_time = sum(r.get('execution_time', 0) for r in results) / len(results)
        avg_actions = sum(r.get('num_actions', 0) for r in results) / len(results)
        
        if 'reward' in results[0]:
            avg_reward = sum(r.get('reward', 0) for r in results) / len(results)
        else:
            avg_reward = 0
        
        return {
            'total_tasks': len(results),
            'successful_tasks': len(successful),
            'success_rate': success_rate,
            'avg_execution_time': avg_time,
            'avg_actions_per_task': avg_actions,
            'avg_reward': avg_reward
        }
    
    def print_comparison(self):
        """Print comparison table"""
        original_metrics = self.compute_metrics(self.original_results)
        extended_metrics = self.compute_metrics(self.extended_results)
        
        table = Table(title="Original vs Extended RaDA Comparison")
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Original", style="blue")
        table.add_column("Extended", style="green")
        table.add_column("Improvement", style="yellow")
        
        metrics_to_compare = [
            ("Success Rate", "success_rate", "percentage"),
            ("Avg Execution Time", "avg_execution_time", "time"),
            ("Avg Actions", "avg_actions_per_task", "number"),
            ("Avg Reward", "avg_reward", "number")
        ]
        
        for metric_name, key, fmt in metrics_to_compare:
            orig_val = original_metrics.get(key, 0)
            ext_val = extended_metrics.get(key, 0)
            
            if fmt == "percentage":
                orig_str = f"{orig_val:.2%}"
                ext_str = f"{ext_val:.2%}"
                improvement = ((ext_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
                imp_str = f"{improvement:+.1f}%" if improvement != 0 else "-"
            elif fmt == "time":
                orig_str = f"{orig_val:.2f}s"
                ext_str = f"{ext_val:.2f}s"
                improvement = ((orig_val - ext_val) / orig_val * 100) if orig_val > 0 else 0
                imp_str = f"{improvement:+.1f}%" if improvement != 0 else "-"
            else:
                orig_str = f"{orig_val:.3f}"
                ext_str = f"{ext_val:.3f}"
                improvement = ((ext_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
                imp_str = f"{improvement:+.1f}%" if improvement != 0 else "-"
            
            table.add_row(metric_name, orig_str, ext_str, imp_str)
        
        console.print(table)
    
    def save_results(self):
        """Save comparison results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        comparison_data = {
            'timestamp': timestamp,
            'original_results': self.original_results,
            'extended_results': self.extended_results,
            'original_metrics': self.compute_metrics(self.original_results),
            'extended_metrics': self.compute_metrics(self.extended_results)
        }
        
        output_file = self.output_dir / f"comparison_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info(f"Comparison results saved to {output_file}")
        return output_file


async def main():
    """Main evaluation function"""
    console.print("="*80)
    console.print(Panel(
        "[bold magenta]ORIGINAL vs EXTENDED RaDA COMPARISON[/bold magenta]\n"
        "Extended Features:\n"
        "• Adaptive Memory (Lifelong Retrieval-Augmented Planning)\n"
        "• RL + RaDA Learning Loop\n"
        "• Graph-based Task Decomposition",
        border_style="magenta"
    ))
    console.print("="*80 + "\n")
    
    evaluator = ComparisonEvaluator()
    
    # Load dataset
    console.print("[yellow]Loading dataset...[/yellow]")
    tasks = load_dataset("mind2web")
    console.print(f"Loaded {len(tasks)} tasks\n")
    
    # Evaluate original
    original_results = await evaluator.evaluate_original(tasks, max_tasks=3)
    
    # Evaluate extended (with multiple episodes for RL learning)
    extended_results = await evaluator.evaluate_extended(tasks, max_tasks=3, num_episodes=2)
    
    # Print comparison
    console.print("\n")
    evaluator.print_comparison()
    
    # Save results
    evaluator.save_results()
    
    console.print("\n" + "="*80)
    console.print("[bold green]✓ Comparison Complete[/bold green]")
    console.print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
