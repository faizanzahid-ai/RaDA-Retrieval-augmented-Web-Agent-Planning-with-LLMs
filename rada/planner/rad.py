"""
RaD: Retrieval-augmented Task Decomposition (Planner)

Decomposes complex web tasks into high-level subtasks using LLM
with retrieval augmentation from similar tasks
Paper-specific implementation from RaDA ACL 2024
"""

import json
from typing import List, Dict, Any, Optional
from loguru import logger

from rada.utils.groq_client import GroqClient
from rada.utils.prompts import (
    COMPWOB_RAD_SUBTASK_DECOMPOSITION,
    COMPWOB_RAD_SUBTASK_PLANNING,
    MIND2WEB_RAD_SUBTASK_DECOMPOSITION,
    compose_subtask_query
)


class RaDPlanner:
    """
    Task decomposition planner with retrieval augmentation
    
    Implements the RaD component from the RaDA paper (Algorithm 1, line 7)
    """
    
    def __init__(
        self,
        llm_model: str = None,
        llm_client=None,
        api_key: Optional[str] = None,
        benchmark: str = "compwob"
    ):
        self.benchmark = benchmark
        
        # Support both llm_client (new) and llm_model (legacy)
        if llm_client is not None:
            self.client = llm_client
            self.llm_model = getattr(llm_client, 'model', 'unknown')
        else:
            from rada.utils.groq_client import GroqClient
            self.llm_model = llm_model or "llama-3.3-70b-versatile"
            self.client = GroqClient(model=self.llm_model, api_key=api_key)
        
        logger.info(f"RaDPlanner initialized with {self.llm_model} for {benchmark}")
    
    def should_decompose(
        self,
        task: str,
        min_exemplar_length: int = 10
    ) -> bool:
        """
        Decomposition trigger logic from paper:
        - CompWoB: always triggered
        - Mind2Web: triggered if task length exceeds minimum exemplar by 50%
        
        Args:
            task: Task description
            min_exemplar_length: Minimum exemplar length for comparison
            
        Returns:
            Whether to trigger decomposition
        """
        if self.benchmark == "compwob":
            # CompWoB: decomposition behavior is always triggered
            return True
        elif self.benchmark == "mind2web":
            # Mind2Web: triggered if task length exceeds minimum by 50%
            task_length = len(task.split())
            threshold = min_exemplar_length * 1.5
            return task_length > threshold
        else:
            # Default: always decompose
            return True
    
    def decompose(
        self,
        task: str,
        url: str,
        page_title: str,
        page_summary: str,
        similar_tasks: Optional[List[Dict]] = None,
        min_exemplar_length: int = 10
    ) -> List[str]:
        """
        Decompose task into subtasks using paper-specific prompts
        
        Args:
            task: Task description
            url: Current URL
            page_title: Page title
            page_summary: Page content summary
            similar_tasks: Retrieved similar tasks
            min_exemplar_length: Minimum exemplar length for trigger logic
            
        Returns:
            List of subtasks
        """
        # Check decomposition trigger
        if not self.should_decompose(task, min_exemplar_length):
            logger.info("Decomposition not triggered, returning single subtask")
            return [task]
        
        # Build retrieved exemplars string
        exemplars_str = self._format_exemplars(similar_tasks)
        
        # Select prompt based on benchmark
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
        
        subtasks = self._parse_subtasks_paper_format(response)
        logger.info(f"Decomposed task into {len(subtasks)} subtasks")
        
        return subtasks
    
    def plan_subtasks(
        self,
        task: str,
        subtasks: List[str],
        similar_tasks: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Plan subtask ordering (RaD Subtask Planning from paper)
        
        Args:
            task: Task description
            subtasks: Unordered subtasks
            similar_tasks: Retrieved similar tasks
            
        Returns:
            Ordered list of subtasks
        """
        exemplars_str = self._format_exemplars(similar_tasks)
        subtasks_str = "\n".join([f"[Subtask] {st}" for st in subtasks])
        
        prompt = COMPWOB_RAD_SUBTASK_PLANNING.format(
            retrieved_exemplars=exemplars_str,
            subtasks=subtasks_str,
            task=task
        )
        
        response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a large language model trained to navigate the web."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
        
        ordered_subtasks = self._parse_ordered_subtasks(response)
        logger.info(f"Planned {len(ordered_subtasks)} ordered subtasks")
        
        return ordered_subtasks
    
    def _format_exemplars(self, similar_tasks: Optional[List[Dict]]) -> str:
        """Format retrieved exemplars for prompt"""
        if not similar_tasks:
            return "No exemplars available."
        
        exemplars = []
        for i, task in enumerate(similar_tasks[:3], 1):
            exemplar = f"Exemplar {i}:\n"
            exemplar += f"Task: {task.get('task', 'N/A')}\n"
            if 'subtasks' in task:
                exemplar += f"Subtasks: {task['subtasks']}\n"
            if 'trajectory' in task:
                exemplar += f"Trajectory: {task['trajectory'][:200]}...\n"
            exemplars.append(exemplar)
        
        return "\n".join(exemplars)
    
    def _parse_subtasks_paper_format(self, response: str) -> List[str]:
        """
        Parse subtasks from paper format response
        Format: [Subtask] description
        Ends with EOS
        """
        subtasks = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('[Subtask]'):
                subtask = line.replace('[Subtask]', '').strip()
                # Clean up common LLM artifacts
                subtask = subtask.replace('newline', '').strip()
                subtask = subtask.rstrip('\\n').strip()
                if subtask:
                    subtasks.append(subtask)
            elif line == 'EOS':
                break
        
        if not subtasks:
            # Fallback: try JSON parsing
            try:
                start = response.find('[')
                end = response.rfind(']') + 1
                if start != -1 and end > start:
                    json_str = response[start:end]
                    subtasks = json.loads(json_str)
                    if isinstance(subtasks, list):
                        # Clean each subtask
                        subtasks = [s.replace('newline', '').strip() for s in subtasks if s and s.strip()]
                        return subtasks
            except:
                pass
            
            # Final fallback: split by newlines
            subtasks = []
            for line in lines:
                cleaned = line.strip().replace('newline', '').strip()
                if cleaned and cleaned != 'EOS':
                    subtasks.append(cleaned)
        
        return subtasks
    
    def _parse_ordered_subtasks(self, response: str) -> List[str]:
        """
        Parse ordered subtasks from paper format
        Format: 1. [Subtask] description. 2. [Subtask] description...
        Ends with EOS
        """
        subtasks = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line == 'EOS':
                break
            # Remove numbering (1., 2., etc.)
            if line and line[0].isdigit():
                # Remove "N. " prefix
                parts = line.split('.', 1)
                if len(parts) > 1:
                    subtask = parts[1].strip()
                    # Remove [Subtask] prefix if present
                    if subtask.startswith('[Subtask]'):
                        subtask = subtask.replace('[Subtask]', '').strip()
                    if subtask:
                        subtasks.append(subtask)
        
        if not subtasks:
            # Fallback
            subtasks = [line.strip() for line in lines if line.strip() and line != 'EOS']
        
        return subtasks
