"""CompWoB Evaluator"""

from typing import List, Dict
from loguru import logger


class CompWoBEvaluator:
    """Evaluate on CompWoB benchmark"""
    
    def evaluate(self, results: List[Dict]) -> Dict:
        """Evaluate CompWoB results"""
        success_rate = sum(1 for r in results if r.get('success')) / len(results) if results else 0
        
        metrics = {
            'dataset': 'compwob',
            'total_tasks': len(results),
            'success_rate': success_rate,
            'avg_actions': sum(r.get('total_actions', 0) for r in results) / len(results) if results else 0,
            'avg_time': sum(r.get('execution_time', 0) for r in results) / len(results) if results else 0
        }
        
        logger.info(f"CompWoB Success Rate: {success_rate:.2%}")
        return metrics
