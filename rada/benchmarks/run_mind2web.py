"""Run Mind2Web benchmark"""

import asyncio
from typing import Dict
from loguru import logger
from rich.console import Console

from rada.data.mind2web.dataset import Mind2WebDataset
from rada.evaluation.mind2web_evaluator import Mind2WebEvaluator
from rada.main import RADA

console = Console()


async def run_mind2web(max_tasks: int = None, return_results: bool = False):
    """Run Mind2Web benchmark"""
    console.print("[bold blue]Running Mind2Web Benchmark[/bold blue]")
    
    # Load dataset
    dataset = Mind2WebDataset()
    tasks = dataset.get_tasks()
    if max_tasks:
        tasks = tasks[:max_tasks]
    
    logger.info(f"Evaluating on {len(tasks)} Mind2Web tasks")
    
    # Initialize RaDA
    agent = RADA()
    
    # Run evaluation
    results = []
    for task in tasks:
        result = await agent.execute_task(
            task=task['task'],
            start_url=task['start_url']
        )
        result['task_id'] = task['task_id']
        result['subtasks_total'] = len(task.get('subtasks', []))
        result['difficulty'] = task.get('difficulty', 'unknown')
        result['domain'] = task.get('domain', 'unknown')
        results.append(result)
    
    # Evaluate
    evaluator = Mind2WebEvaluator()
    ground_truths = [{'steps': task.get('subtasks', []), 'actions': task.get('actions', []), 'elements': task.get('elements', []), 'operations': task.get('operations', [])} for task in tasks]
    metrics = evaluator.evaluate_dataset(results, ground_truths, 'mind2web')
    
    if return_results:
        return metrics, results
    return metrics
