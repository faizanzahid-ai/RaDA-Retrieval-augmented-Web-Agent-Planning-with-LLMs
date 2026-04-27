"""
Graph-based RaD: Graph-structured Task Decomposition

Implements graph-based task decomposition where:
- Nodes = subtasks
- Edges = dependencies
- Enables parallel execution of independent subtasks
- Better planning through dependency awareness
"""

import json
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

from rada.utils.groq_client import GroqClient
from rada.utils.prompts import (
    COMPWOB_RAD_SUBTASK_DECOMPOSITION,
    MIND2WEB_RAD_SUBTASK_DECOMPOSITION,
    compose_subtask_query
)


@dataclass
class SubtaskNode:
    """Represents a subtask node in the graph"""
    id: str
    description: str
    dependencies: List[str]  # List of subtask IDs this depends on
    completed: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TaskGraph:
    """Graph-structured task decomposition"""
    subtasks: List[SubtaskNode]
    edges: List[Tuple[str, str]]  # (from_id, to_id) representing dependencies
    
    def to_dict(self) -> Dict:
        return {
            "subtasks": [s.to_dict() for s in self.subtasks],
            "edges": self.edges
        }
    
    def get_ready_subtasks(self) -> List[SubtaskNode]:
        """Get subtasks that are ready to execute (all dependencies met)"""
        ready = []
        for subtask in self.subtasks:
            if not subtask.completed:
                # Check if all dependencies are completed
                deps_met = all(
                    any(s.id == dep_id and s.completed for s in self.subtasks)
                    for dep_id in subtask.dependencies
                )
                if deps_met:
                    ready.append(subtask)
        return ready
    
    def get_parallel_groups(self) -> List[List[SubtaskNode]]:
        """
        Get groups of subtasks that can be executed in parallel
        
        Returns:
            List of groups, where each group contains subtasks that can be executed together
        """
        groups = []
        remaining = [s for s in self.subtasks if not s.completed]
        
        while remaining:
            # Find subtasks with no unmet dependencies
            ready = []
            for subtask in remaining:
                deps_met = all(
                    any(s.id == dep_id and s.completed for s in self.subtasks)
                    for dep_id in subtask.dependencies
                )
                if deps_met:
                    ready.append(subtask)
            
            if not ready:
                # Circular dependency or all completed
                break
            
            groups.append(ready)
            remaining = [s for s in remaining if s not in ready]
        
        return groups


class GraphRaDPlanner:
    """
    Graph-based task decomposition planner
    
    Decomposes tasks into graph-structured subtasks with dependency tracking
    """
    
    def __init__(
        self,
        llm_model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        benchmark: str = "compwob"
    ):
        self.llm_model = llm_model
        self.benchmark = benchmark
        self.client = GroqClient(model=llm_model, api_key=api_key)
        logger.info(f"GraphRaDPlanner initialized with {llm_model} for {benchmark}")
    
    def decompose(
        self,
        task: str,
        url: str,
        page_title: str,
        page_summary: str,
        similar_tasks: Optional[List[Dict]] = None
    ) -> TaskGraph:
        """
        Decompose task into graph-structured subtasks
        
        Args:
            task: Task description
            url: Current URL
            page_title: Page title
            page_summary: Page summary
            similar_tasks: Similar tasks for retrieval augmentation
            
        Returns:
            TaskGraph with subtasks and dependencies
        """
        # First, get linear subtasks using original decomposition
        linear_subtasks = self._get_linear_subtasks(
            task, url, page_title, page_summary, similar_tasks
        )
        
        # Then, analyze and add dependencies using LLM
        graph = self._analyze_dependencies(linear_subtasks, task)
        
        logger.info(f"Decomposed task into {len(graph.subtasks)} subtasks with {len(graph.edges)} dependencies")
        return graph
    
    def _get_linear_subtasks(
        self,
        task: str,
        url: str,
        page_title: str,
        page_summary: str,
        similar_tasks: Optional[List[Dict]] = None
    ) -> List[str]:
        """Get linear subtasks using original RaD method"""
        # Format exemplars
        exemplars_str = "No exemplars available."
        if similar_tasks:
            exemplars = []
            for i, t in enumerate(similar_tasks[:3], 1):
                exemplar = f"Exemplar {i}:\n"
                exemplar += f"Task: {t.get('task', 'N/A')}\n"
                if 'subtasks' in t:
                    exemplar += f"Subtasks: {t['subtasks']}\n"
                exemplars.append(exemplar)
            exemplars_str = '\n'.join(exemplars)
        
        if self.benchmark == "compwob":
            prompt = COMPWOB_RAD_SUBTASK_DECOMPOSITION.format(
                retrieved_exemplars=exemplars_str,
                task=task
            )
            system_prompt = "You are a large language model trained to navigate the web."
        else:  # mind2web
            prompt = MIND2WEB_RAD_SUBTASK_DECOMPOSITION.format(
                retrieved_exemplars=exemplars_str,
                task=task
            )
            system_prompt = "You are a large language model trained to navigate the web."
        
        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        # Parse subtasks from response
        subtasks = self._parse_subtasks(response)
        return subtasks
    
    def _parse_subtasks(self, response: str) -> List[str]:
        """Parse subtasks from LLM response"""
        # Try to extract numbered list
        import re
        matches = re.findall(r'\d+\.\s*(.+?)(?=\n|$)', response)
        if matches:
            return [m.strip() for m in matches]
        
        # Fallback: split by newlines
        lines = response.strip().split('\n')
        subtasks = [line.strip() for line in lines if line.strip() and not line.strip().isdigit()]
        return subtasks
    
    def _analyze_dependencies(self, subtasks: List[str], task: str) -> TaskGraph:
        """
        Analyze dependencies between subtasks using LLM
        
        Args:
            subtasks: List of linear subtasks
            task: Original task description
            
        Returns:
            TaskGraph with dependencies
        """
        # Create subtask nodes
        nodes = [
            SubtaskNode(
                id=f"subtask_{i}",
                description=subtask,
                dependencies=[]
            )
            for i, subtask in enumerate(subtasks)
        ]
        
        # Use LLM to identify dependencies
        prompt = f"""
Analyze the following task and its subtasks. Identify which subtasks depend on others.

Task: {task}

Subtasks:
{chr(10).join(f"{i+1}. {subtask}" for i, subtask in enumerate(subtasks))}

Output format: For each subtask, list the numbers of subtasks it depends on (0 if none).
Example:
1. Depends on: 0
2. Depends on: 1
3. Depends on: 0
4. Depends on: 2,3

Analyze dependencies:
"""
        
        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a task planning expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        # Parse dependencies
        edges = []
        for i, node in enumerate(nodes):
            deps = self._parse_dependencies_for_subtask(response, i + 1)
            node.dependencies = [f"subtask_{d-1}" for d in deps if d > 0]
            # Create edges
            for dep in node.dependencies:
                edges.append((dep, node.id))
        
        return TaskGraph(subtasks=nodes, edges=edges)
    
    def _parse_dependencies_for_subtask(self, response: str, subtask_num: int) -> List[int]:
        """Parse dependencies for a specific subtask"""
        import re
        pattern = rf'{subtask_num}\.\s*Depends on:\s*(\d+(?:,\s*\d+)*)'
        match = re.search(pattern, response, re.IGNORECASE)
        
        if match:
            deps_str = match.group(1)
            deps = [int(d.strip()) for d in deps_str.split(',') if d.strip().isdigit()]
            return deps
        
        # Default: depends on previous subtask
        if subtask_num > 1:
            return [subtask_num - 1]
        return []
    
    def get_execution_order(self, graph: TaskGraph) -> List[List[SubtaskNode]]:
        """
        Get optimal execution order with parallelization
        
        Args:
            graph: Task graph
            
        Returns:
            List of groups, each group can be executed in parallel
        """
        return graph.get_parallel_groups()
    
    def mark_completed(self, graph: TaskGraph, subtask_id: str):
        """Mark a subtask as completed"""
        for subtask in graph.subtasks:
            if subtask.id == subtask_id:
                subtask.completed = True
                break
