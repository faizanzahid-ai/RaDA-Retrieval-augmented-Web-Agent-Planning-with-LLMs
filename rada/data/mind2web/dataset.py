"""
Mind2Web Dataset

Dataset structure from the RaDA paper with tasks across multiple domains
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger


class Mind2WebDataset:
    """Mind2Web dataset loader and manager"""
    
    def __init__(self, data_path: str = "./data/mind2web"):
        self.data_path = Path(data_path)
        self.tasks = self._load_tasks()
    
    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from dataset"""
        if not self.data_path.exists():
            logger.info("Mind2Web data directory not found, creating sample tasks")
            return self._create_sample_tasks()
        
        tasks_file = self.data_path / "tasks.json"
        if tasks_file.exists():
            with open(tasks_file, 'r') as f:
                return json.load(f)['tasks']
        
        return self._create_sample_tasks()
    
    def _create_sample_tasks(self) -> List[Dict[str, Any]]:
        """Create sample Mind2Web tasks with fast, accessible websites"""
        return [
            {
                "task_id": "mind2web_001",
                "domain": "Information",
                "website": "Wikipedia",
                "task": "Search for information about artificial intelligence",
                "start_url": "https://en.wikipedia.org",
                "subtasks": [
                    "Navigate to search box",
                    "Enter 'artificial intelligence'",
                    "Click search button",
                    "View results"
                ],
                "difficulty": "easy",
                "success_criteria": "Wikipedia page about AI is displayed"
            },
            {
                "task_id": "mind2web_002",
                "domain": "Testing",
                "website": "Example.com",
                "task": "Navigate to example.com and verify page loads",
                "start_url": "https://example.com",
                "subtasks": [
                    "Navigate to example.com",
                    "Verify page title",
                    "Check page content"
                ],
                "difficulty": "easy",
                "success_criteria": "Example.com page loads successfully"
            },
            {
                "task_id": "mind2web_003",
                "domain": "Information",
                "website": "Wikipedia",
                "task": "Find information about machine learning on Wikipedia",
                "start_url": "https://en.wikipedia.org",
                "subtasks": [
                    "Navigate to search box",
                    "Enter 'machine learning'",
                    "Click search",
                    "View article"
                ],
                "difficulty": "easy",
                "success_criteria": "Machine learning Wikipedia article is displayed"
            }
        ]
    
    def get_tasks(self, difficulty: str = None, domain: str = None) -> List[Dict[str, Any]]:
        """Get tasks with optional filters"""
        tasks = self.tasks
        
        if difficulty:
            tasks = [t for t in tasks if t.get('difficulty') == difficulty]
        
        if domain:
            tasks = [t for t in tasks if t.get('domain', '').lower() == domain.lower()]
        
        return tasks
    
    def get_task_by_id(self, task_id: str) -> Dict[str, Any]:
        """Get specific task by ID"""
        for task in self.tasks:
            if task['task_id'] == task_id:
                return task
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        difficulty_counts = {}
        domain_counts = {}
        
        for task in self.tasks:
            diff = task.get('difficulty', 'unknown')
            domain = task.get('domain', 'unknown')
            
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return {
            'total_tasks': len(self.tasks),
            'difficulty_distribution': difficulty_counts,
            'domain_distribution': domain_counts
        }
