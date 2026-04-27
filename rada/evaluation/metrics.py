"""
Evaluation metrics for RaDA
"""

from typing import List, Dict, Any
from loguru import logger


class TaskSuccessRate:
    """Calculate task success rate"""
    
    def __init__(self):
        self.total_tasks = 0
        self.successful_tasks = 0
    
    def update(self, task_result: Dict[str, Any]):
        """
        Update metrics with task result
        
        Args:
            task_result: Task execution result
        """
        self.total_tasks += 1
        if task_result.get("success", False):
            self.successful_tasks += 1
    
    def get_success_rate(self) -> float:
        """Get success rate"""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats"""
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "success_rate": self.get_success_rate()
        }


class ActionEfficiency:
    """Calculate action efficiency metrics"""
    
    def __init__(self):
        self.total_actions = 0
        self.total_tasks = 0
        self.action_counts = []
    
    def update(self, task_result: Dict[str, Any]):
        """
        Update metrics with task result
        
        Args:
            task_result: Task execution result
        """
        self.total_tasks += 1
        num_actions = task_result.get("total_actions", 0)
        self.total_actions += num_actions
        self.action_counts.append(num_actions)
    
    def get_average_actions(self) -> float:
        """Get average number of actions per task"""
        if self.total_tasks == 0:
            return 0.0
        return self.total_actions / self.total_tasks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats"""
        return {
            "total_tasks": self.total_tasks,
            "total_actions": self.total_actions,
            "average_actions_per_task": self.get_average_actions(),
            "action_counts": self.action_counts
        }


class SubtaskCompletionRate:
    """Calculate subtask completion rate"""
    
    def __init__(self):
        self.total_subtasks = 0
        self.completed_subtasks = 0
    
    def update(self, subtask_results: List[Dict[str, Any]]):
        """
        Update metrics with subtask results
        
        Args:
            subtask_results: List of subtask execution results
        """
        for result in subtask_results:
            self.total_subtasks += 1
            if result.get("success", False):
                self.completed_subtasks += 1
    
    def get_completion_rate(self) -> float:
        """Get subtask completion rate"""
        if self.total_subtasks == 0:
            return 0.0
        return self.completed_subtasks / self.total_subtasks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats"""
        return {
            "total_subtasks": self.total_subtasks,
            "completed_subtasks": self.completed_subtasks,
            "completion_rate": self.get_completion_rate()
        }


class BenchmarkEvaluator:
    """Evaluate agent performance on benchmarks"""
    
    def __init__(self):
        self.task_success = TaskSuccessRate()
        self.action_efficiency = ActionEfficiency()
        self.subtask_completion = SubtaskCompletionRate()
    
    def evaluate_task(self, task_result: Dict[str, Any]):
        """
        Evaluate a single task result
        
        Args:
            task_result: Task execution result
        """
        self.task_success.update(task_result)
        self.action_efficiency.update(task_result)
        
        logger.info(f"Task evaluated: Success={task_result.get('success', False)}")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation metrics"""
        return {
            "task_success": self.task_success.get_stats(),
            "action_efficiency": self.action_efficiency.get_stats(),
            "subtask_completion": self.subtask_completion.get_stats()
        }
    
    def reset(self):
        """Reset all metrics"""
        self.task_success = TaskSuccessRate()
        self.action_efficiency = ActionEfficiency()
        self.subtask_completion = SubtaskCompletionRate()
        logger.info("Metrics reset")
