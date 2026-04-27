"""
Verifier: Subtask Completion Verification

Verifies whether a subtask has been successfully completed
using LLM-based state analysis
Paper-specific implementation from RaDA ACL 2024
"""

import json
from typing import Dict, Any, Optional, List
from loguru import logger

from rada.utils.groq_client import GroqClient
from rada.utils.prompts import (
    COMPWOB_SUBTASK_VERIFICATION_NEXT,
    MIND2WEB_RAD_SUBTASK_VERIFICATION
)


class SubtaskVerifier:
    """
    Verifies subtask completion
    
    Implements the πV component from the RaDA paper (Algorithm 1, lines 9, 15)
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
    
    def verify(
        self,
        subtask: str,
        page_title: str,
        url: str,
        page_summary: str,
        previous_action: Dict[str, Any],
        subtasks: List[str],
        completed_subtasks: List[str],
        action_history: List[Dict],
        trajectory: str
    ) -> Dict[str, Any]:
        """
        Verify subtask completion using paper-specific prompts
        
        Args:
            subtask: Subtask description
            page_title: Current page title
            url: Current URL
            page_summary: Page state summary
            previous_action: Last action taken
            subtasks: All subtasks
            completed_subtasks: Completed subtasks
            action_history: Action history
            trajectory: Current trajectory
            
        Returns:
            Verification result with completion status and confidence
        """
        # Select prompt based on benchmark
        if self.benchmark == "compwob":
            # CompWoB: Subtask Verification and Next Subtask
            subtasks_str = "\n".join([f"[{i}] {st}" for i, st in enumerate(subtasks)])
            completed_str = "\n".join([f"[{i}] {st}" for i, st in enumerate(completed_subtasks)])
            action_history_str = self._format_action_history(action_history)
            
            prompt = COMPWOB_SUBTASK_VERIFICATION_NEXT.format(
                subtasks=subtasks_str,
                completed_subtasks=completed_str,
                action_history=action_history_str
            )
            system_prompt = "You are a large language model trained to navigate the web."
        else:  # mind2web
            # Mind2Web: Subtask Verification (which subtask was completed)
            remaining_subtasks = [st for st in subtasks if st not in completed_subtasks]
            remaining_str = "\n".join([f"- {st}" for st in remaining_subtasks])
            
            prompt = MIND2WEB_RAD_SUBTASK_VERIFICATION.format(
                remaining_subtasks=remaining_str,
                trajectory=trajectory
            )
            system_prompt = "You are a large language model trained to navigate the web."
        
        response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
        
        return self._parse_verification(response, subtasks)
    
    def get_next_subtask(
        self,
        subtasks: List[str],
        completed_subtasks: List[str],
        action_history: List[Dict]
    ) -> Optional[str]:
        """
        Get the next subtask to execute (CompWoB format)
        
        Args:
            subtasks: All subtasks
            completed_subtasks: Completed subtasks
            action_history: Action history
            
        Returns:
            Next subtask or None
        """
        subtasks_str = "\n".join([f"[{i}] {st}" for i, st in enumerate(subtasks)])
        completed_str = "\n".join([f"[{i}] {st}" for i, st in enumerate(completed_subtasks)])
        action_history_str = self._format_action_history(action_history)
        
        prompt = COMPWOB_SUBTASK_VERIFICATION_NEXT.format(
            subtasks=subtasks_str,
            completed_subtasks=completed_str,
            action_history=action_history_str
        )
        
        response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a large language model trained to navigate the web."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
        
        # Parse [ID: n] format
        import re
        match = re.search(r'\[ID:\s*(\d+)\]', response)
        if match:
            idx = int(match.group(1))
            if idx < len(subtasks):
                return subtasks[idx]
        
        return None
    
    def _parse_verification(self, response: str, subtasks: List[str]) -> Dict[str, Any]:
        """
        Parse verification result
        Handles both paper format (NONE/subtask description) and JSON format
        """
        response = response.strip()
        
        # Mind2Web format: returns "NONE" or subtask description
        if response == "NONE":
            return {
                'completed': False,
                'confidence': 0.0,
                'reasoning': 'No subtask completed',
                'completed_subtask': None
            }
        
        # Check if response matches any subtask
        for subtask in subtasks:
            if subtask.lower() in response.lower():
                return {
                    'completed': True,
                    'confidence': 0.9,
                    'reasoning': f'Subtask matched: {response}',
                    'completed_subtask': subtask
                }
        
        # Try JSON format (legacy)
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                result = json.loads(response[start:end])
                
                # Validate
                if 'completed' not in result:
                    raise ValueError("Missing 'completed' field")
                
                result.setdefault('confidence', 0.0)
                result.setdefault('reasoning', '')
                result.setdefault('completed_subtask', None)
                
                return result
        except:
            pass
        
        # Fallback: assume completed if response contains affirmative words
        affirmative_words = ['yes', 'completed', 'done', 'finished', 'success']
        if any(word in response.lower() for word in affirmative_words):
            return {
                'completed': True,
                'confidence': 0.7,
                'reasoning': response,
                'completed_subtask': None
            }
        
        return {
            'completed': False,
            'confidence': 0.0,
            'reasoning': response,
            'completed_subtask': None
        }
    
    def _format_action_history(self, action_history: List[Dict]) -> str:
        """Format action history for prompt"""
        if not action_history:
            return "No previous actions."
        
        history = []
        for i, action in enumerate(action_history[-5:], 1):  # Last 5 actions
            history.append(f"Action {i}: {action}")
        
        return "\n".join(history)
