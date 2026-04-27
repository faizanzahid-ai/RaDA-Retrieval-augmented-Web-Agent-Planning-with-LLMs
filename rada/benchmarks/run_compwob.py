"""Run CompWoB benchmark"""

import asyncio
from typing import Dict
from loguru import logger

from rada.evaluation.compwob_eval import CompWoBEvaluator
from rada.main import RADA


COMPWOB_TASKS = [
    {
        'task_id': 'compwob_001',
        'task': 'Navigate to Wikipedia and search for Python',
        'start_url': 'https://en.wikipedia.org',
        'subtasks': ['Navigate to search box', 'Enter Python', 'Click search', 'View results']
    },
    {
        'task_id': 'compwob_002',
        'task': 'Visit example.com and verify page loads',
        'start_url': 'https://example.com',
        'subtasks': ['Navigate to example.com', 'Check page title', 'Verify content']
    }
]


async def run_compwob(max_tasks: int = None, return_results: bool = False):
    """Run CompWoB benchmark"""
    tasks = COMPWOB_TASKS[:max_tasks] if max_tasks else COMPWOB_TASKS
    logger.info(f"Evaluating on {len(tasks)} CompWoB tasks")
    
    agent = RADA()
    
    results = []
    for task in tasks:
        result = await agent.execute_task(
            task=task['task'],
            start_url=task['start_url']
        )
        result['task_id'] = task['task_id']
        result['subtasks_total'] = len(task.get('subtasks', []))
        result['task_type'] = task.get('task_type', 'unknown')
        results.append(result)
    
    evaluator = CompWoBEvaluator()
    metrics = evaluator.evaluate(results)
    
    if return_results:
        return metrics, results
    return metrics
