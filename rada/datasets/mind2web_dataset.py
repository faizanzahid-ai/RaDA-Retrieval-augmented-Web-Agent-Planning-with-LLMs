"""
Mind2Web-like Dataset for RaDA Evaluation

This module provides a synthetic dataset inspired by the Mind2Web benchmark
mentioned in the RaDA paper. It includes web navigation tasks across multiple
domains with varying complexity levels.
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from loguru import logger


@dataclass
class DatasetTask:
    """Represents a single task in the evaluation dataset"""
    task_id: str
    domain: str
    website: str
    task_description: str
    start_url: str
    expected_subtasks: List[str]
    difficulty: str  # easy, medium, hard
    success_criteria: str


class Mind2WebDataset:
    """
    Mind2Web-like dataset for evaluating web agents
    
    Based on the RaDA paper which evaluates on Mind2Web and CompWoB benchmarks
    """
    
    def __init__(self, dataset_path: str = "./data/mind2web_sample.json"):
        """
        Initialize dataset
        
        Args:
            dataset_path: Path to dataset JSON file
        """
        self.dataset_path = Path(dataset_path)
        self.tasks: List[DatasetTask] = []
        self._load_or_create_dataset()
    
    def _load_or_create_dataset(self):
        """Load dataset from file or create sample dataset"""
        if self.dataset_path.exists():
            self._load_dataset()
        else:
            logger.info("Dataset file not found, creating sample dataset")
            self._create_sample_dataset()
            self._save_dataset()
    
    def _load_dataset(self):
        """Load dataset from JSON file"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.tasks = [DatasetTask(**task) for task in data['tasks']]
        logger.info(f"Loaded {len(self.tasks)} tasks from dataset")
    
    def _save_dataset(self):
        """Save dataset to JSON file"""
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": {
                "name": "Mind2Web-Sample",
                "version": "1.0",
                "description": "Sample dataset inspired by Mind2Web benchmark",
                "total_tasks": len(self.tasks)
            },
            "tasks": [asdict(task) for task in self.tasks]
        }
        
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.tasks)} tasks to {self.dataset_path}")
    
    def _create_sample_dataset(self):
        """Create a sample dataset with fast, accessible websites"""
        self.tasks = [
            DatasetTask(
                task_id="task_001",
                domain="Information",
                website="Wikipedia",
                task_description="Search for information about artificial intelligence",
                start_url="https://en.wikipedia.org",
                expected_subtasks=[
                    "Navigate to search box",
                    "Enter search query 'artificial intelligence'",
                    "Click search button",
                    "View results"
                ],
                difficulty="easy",
                success_criteria="Wikipedia page about artificial intelligence is displayed"
            ),
            DatasetTask(
                task_id="task_002",
                domain="Testing",
                website="Example.com",
                task_description="Navigate to example.com and verify page loads",
                start_url="https://example.com",
                expected_subtasks=[
                    "Navigate to example.com",
                    "Verify page title",
                    "Check page content"
                ],
                difficulty="easy",
                success_criteria="Example.com page loads successfully"
            ),
            DatasetTask(
                task_id="task_003",
                domain="Information",
                website="Wikipedia",
                task_description="Find information about machine learning on Wikipedia",
                start_url="https://en.wikipedia.org",
                expected_subtasks=[
                    "Navigate to search box",
                    "Enter 'machine learning'",
                    "Click search",
                    "View article"
                ],
                difficulty="easy",
                success_criteria="Machine learning Wikipedia article is displayed"
            ),
            DatasetTask(
                task_id="task_004",
                domain="Testing",
                website="HTTPBin",
                task_description="Test HTTP GET request to httpbin.org",
                start_url="https://httpbin.org",
                expected_subtasks=[
                    "Navigate to httpbin.org",
                    "Click on /get endpoint",
                    "View response"
                ],
                difficulty="easy",
                success_criteria="HTTP GET response is displayed"
            ),
            DatasetTask(
                task_id="task_005",
                domain="Information",
                website="Wikipedia",
                task_description="Search for Python programming language",
                start_url="https://en.wikipedia.org",
                expected_subtasks=[
                    "Navigate to search",
                    "Enter 'Python programming'",
                    "Click search",
                    "View results"
                ],
                difficulty="easy",
                success_criteria="Python programming Wikipedia page is displayed"
            )
        ]
        
        logger.info(f"Created sample dataset with {len(self.tasks)} tasks using fast websites")
    
    def get_tasks_by_difficulty(self, difficulty: str) -> List[DatasetTask]:
        """
        Get tasks filtered by difficulty
        
        Args:
            difficulty: Difficulty level (easy, medium, hard)
            
        Returns:
            List of tasks
        """
        return [task for task in self.tasks if task.difficulty == difficulty]
    
    def get_tasks_by_domain(self, domain: str) -> List[DatasetTask]:
        """
        Get tasks filtered by domain
        
        Args:
            domain: Domain name
            
        Returns:
            List of tasks
        """
        return [task for task in self.tasks if task.domain.lower() == domain.lower()]
    
    def get_all_tasks(self) -> List[DatasetTask]:
        """Get all tasks"""
        return self.tasks
    
    def get_task_by_id(self, task_id: str) -> DatasetTask:
        """
        Get task by ID
        
        Args:
            task_id: Task ID
            
        Returns:
            DatasetTask or None
        """
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        difficulty_counts = {}
        domain_counts = {}
        
        for task in self.tasks:
            difficulty_counts[task.difficulty] = difficulty_counts.get(task.difficulty, 0) + 1
            domain_counts[task.domain] = domain_counts.get(task.domain, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "difficulty_distribution": difficulty_counts,
            "domain_distribution": domain_counts,
            "domains": list(set(task.domain for task in self.tasks))
        }


class CompWoBDataset:
    """
    CompWoB (Complex World of Bits) dataset
    
    Simplified version for evaluation
    """
    
    def __init__(self):
        """Initialize CompWoB dataset"""
        self.tasks = self._create_tasks()
    
    def _create_tasks(self) -> List[DatasetTask]:
        """Create CompWoB-like tasks"""
        return [
            DatasetTask(
                task_id="compwob_001",
                domain="Email",
                website="Gmail",
                task_description="Compose and send an email to test@example.com with subject 'Hello'",
                start_url="https://mail.google.com",
                expected_subtasks=[
                    "Click compose button",
                    "Enter recipient email",
                    "Enter subject line",
                    "Type email body",
                    "Click send button"
                ],
                difficulty="medium",
                success_criteria="Email sent successfully to recipient"
            ),
            DatasetTask(
                task_id="compwob_002",
                domain="Calendar",
                website="Google Calendar",
                task_description="Create a meeting event for tomorrow at 2 PM",
                start_url="https://calendar.google.com",
                expected_subtasks=[
                    "Click create event",
                    "Enter event title",
                    "Set date to tomorrow",
                    "Set time to 2:00 PM",
                    "Save event"
                ],
                difficulty="medium",
                success_criteria="Event created in calendar for tomorrow at 2 PM"
            ),
            DatasetTask(
                task_id="compwob_003",
                domain="Cloud Storage",
                website="Google Drive",
                task_description="Upload a file and share it with view access",
                start_url="https://drive.google.com",
                expected_subtasks=[
                    "Click upload button",
                    "Select file to upload",
                    "Wait for upload to complete",
                    "Right-click file and select share",
                    "Set permission to 'Anyone with link can view'",
                    "Copy share link"
                ],
                difficulty="hard",
                success_criteria="File uploaded and shared with view access"
            )
        ]
    
    def get_all_tasks(self) -> List[DatasetTask]:
        """Get all tasks"""
        return self.tasks


def load_dataset(dataset_name: str = "mind2web") -> List[DatasetTask]:
    """
    Load dataset by name
    
    Args:
        dataset_name: Dataset name (mind2web or compwob)
        
    Returns:
        List of tasks
    """
    if dataset_name.lower() == "mind2web":
        dataset = Mind2WebDataset()
        return dataset.get_all_tasks()
    elif dataset_name.lower() == "compwob":
        dataset = CompWoBDataset()
        return dataset.get_all_tasks()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
