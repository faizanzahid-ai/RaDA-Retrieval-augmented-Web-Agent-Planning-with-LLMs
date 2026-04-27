"""
RaA: Retrieval-augmented Action Generation (Executor)

Generates specific browser actions based on subtasks and retrieved exemplars
with DOM element grounding
Paper-specific implementation from RaDA ACL 2024
"""

import json
from typing import Dict, Any, Optional, List
from loguru import logger

from rada.utils.groq_client import GroqClient
from rada.utils.prompts import (
    COMPWOB_RAA_ACTION_GENERATION,
    MIND2WEB_RAA_ACTION_GENERATION,
    compose_subtask_query
)


class RaAExecutor:
    """
    Action generator with retrieval augmentation
    
    Implements the RaA component from the RaDA paper (Algorithm 1, line 10)
    Uses subtask-compositional query QT = γ1 ⊕ γ2 ⊕ ... ⊕ γm (Eq. 3)
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
        
        logger.info(f"RaAExecutor initialized with {self.llm_model} for {benchmark}")
    
    def generate_action(
        self,
        task: str,
        subtask: str,
        remaining_subtasks: List[str],
        trajectory: str,
        action_history: List[Dict],
        retrieved_exemplars: str
    ) -> Dict[str, Any]:
        """
        Generate next action using paper-specific prompts
        
        Args:
            task: Overall task
            subtask: Current subtask
            remaining_subtasks: List of remaining subtasks
            trajectory: Current trajectory/state
            action_history: Actions already taken
            retrieved_exemplars: Retrieved exemplars for subtask
            
        Returns:
            Action dictionary
        """
        # Compose subtask-compositional query QT (Eq. 3)
        # CompWoB: single subtask, Mind2Web: all remaining subtasks
        query_type = "single" if self.benchmark == "compwob" else "all"
        QT = compose_subtask_query(remaining_subtasks, query_type=query_type)
        
        # Format action history
        action_history_str = self._format_action_history(action_history)
        
        # Select prompt based on benchmark
        if self.benchmark == "compwob":
            prompt = COMPWOB_RAA_ACTION_GENERATION.format(
                retrieved_exemplars=retrieved_exemplars,
                trajectory=trajectory,
                action_history=action_history_str,
                subtask_description=subtask
            )
            system_prompt = "You are a large language model trained to navigate the web."
        else:  # mind2web
            remaining_str = "\n".join([f"- {st}" for st in remaining_subtasks])
            prompt = MIND2WEB_RAA_ACTION_GENERATION.format(
                retrieved_exemplars=retrieved_exemplars,
                remaining_subtasks=remaining_str,
                trajectory=trajectory
            )
            system_prompt = "You are a large language model trained to navigate the web."
        
        response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
        
        action = self._parse_action(response)
        return self._ensure_action_validity(action)
    
    def _format_action_history(self, action_history: List[Dict]) -> str:
        """Format action history for prompt"""
        if not action_history:
            return "No previous actions."
        
        history = []
        for i, action in enumerate(action_history[-5:], 1):  # Last 5 actions
            history.append(f"Action {i}: {action}")
        
        return "\n".join(history)
    
    def _parse_action(self, response: str) -> Dict[str, Any]:
        """
        Parse action from response
        Handles both paper format (CompWoB/Mind2Web), natural language, and JSON format
        """
        import re
        response_lower = response.lower()
        
        # Try paper format for CompWoB (function calls)
        if 'type(' in response or 'click_xpath(' in response or 'press(' in response:
            return self._parse_compwob_action(response)
        
        # Try paper format for Mind2Web (CLICK/TYPE/SELECT commands)
        first_line = response.strip().split('\n')[0].strip()
        if first_line.startswith('CLICK') or first_line.startswith('TYPE') or first_line.startswith('SELECT'):
            return self._parse_mind2web_action(first_line)
        
        # Try to extract JSON action - improved regex
        json_patterns = [
            r'\{[^{}]*"action"\s*:\s*"[^"]+"[^{}]*\}',  # Simple JSON
            r'\{[^{}]*\'action\'\s*:\s*\'[^"]+\'[^{}]*\}',  # Single-quoted
        ]
        for pattern in json_patterns:
            json_matches = re.findall(pattern, response, re.IGNORECASE)
            if json_matches:
                try:
                    action = json.loads(json_matches[0])
                    if action.get('action') == 'INPUT':
                        action['action'] = 'TYPE'
                    action.setdefault('element_id', None)
                    # Convert None/null value to empty string for TYPE actions
                    if action.get('action') == 'TYPE' and not action.get('value'):
                        action['value'] = ''
                    elif 'value' not in action:
                        action['value'] = None
                    action.setdefault('reasoning', response[:200])
                    return action
                except (json.JSONDecodeError, KeyError):
                    pass
        
        # Try broader JSON extraction
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                candidate = response[start:end]
                action = json.loads(candidate)
                if isinstance(action, dict) and 'action' in action:
                    if action.get('action') == 'INPUT':
                        action['action'] = 'TYPE'
                    action.setdefault('element_id', None)
                    if action.get('action') == 'TYPE' and not action.get('value'):
                        action['value'] = ''
                    elif 'value' not in action:
                        action['value'] = None
                    action.setdefault('reasoning', '')
                    return action
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        
        # Try natural language parsing
        if any(word in response_lower for word in ['click', 'click on', 'select']):
            return {
                'action': 'CLICK',
                'element_id': None,
                'value': None,
                'reasoning': response[:300]
            }
        elif any(word in response_lower for word in ['type', 'enter', 'input']):
            # Try to extract text to type
            text_pattern = r'["\']([^"\']{1,100})["\']'
            text_match = re.search(text_pattern, response)
            text = text_match.group(1) if text_match else ''
            
            return {
                'action': 'TYPE',
                'element_id': None,
                'value': text[:100] if text else '',
                'reasoning': response[:300]
            }
        elif any(word in response_lower for word in ['search', 'navigate', 'go to']):
            # Try to extract URL from response
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls = re.findall(url_pattern, response)
            url = urls[0] if urls else 'https://www.google.com'
            
            return {
                'action': 'NAVIGATE',
                'element_id': None,
                'value': url,
                'reasoning': response[:300]
            }
        elif any(word in response_lower for word in ['press', 'hit', 'submit']):
            # Try to extract key
            for key in ['enter', 'tab', 'escape', 'space', 'backspace']:
                if key in response_lower:
                    return {
                        'action': 'PRESS',
                        'element_id': None,
                        'value': key,
                        'reasoning': response[:300]
                    }
            return {
                'action': 'PRESS',
                'element_id': None,
                'value': 'Enter',
                'reasoning': response[:300]
            }
        
        # Final fallback
        logger.warning(f"Could not parse action, using WAIT: {response[:100]}")
        return {
            'action': 'WAIT',
            'element_id': None,
            'value': '1000',
            'reasoning': response[:300]
        }
    
    def _ensure_action_validity(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure action has all required fields"""
        if action.get('action') == 'TYPE':
            if not action.get('element_id'):
                action['element_id'] = None  # Let grounding handle it
            if action.get('value') is None:
                action['value'] = ''
        elif action.get('action') == 'CLICK':
            if not action.get('element_id'):
                action['element_id'] = None  # Let grounding handle it
        elif action.get('action') == 'NAVIGATE':
            if not action.get('value'):
                action['value'] = 'https://www.google.com'
        return action
    
    def _parse_compwob_action(self, response: str) -> Dict[str, Any]:
        """Parse CompWoB action format (function calls)"""
        response = response.strip()
        
        if 'type(' in response:
            # Extract type argument
            import re
            match = re.search(r'type\(["\'](.+?)["\']\)', response)
            if match:
                return {
                    'action': 'TYPE',
                    'element_id': 'search_box',  # Default element_id
                    'value': match.group(1),
                    'reasoning': response
                }
        elif 'click_xpath(' in response:
            import re
            match = re.search(r'click_xpath\(["\'](.+?)["\']\)', response)
            if match:
                return {
                    'action': 'CLICK',
                    'element_id': match.group(1) or 'button',
                    'value': None,
                    'reasoning': response
                }
        elif 'press(' in response:
            import re
            match = re.search(r'press\(["\'](.+?)["\']\)', response)
            if match:
                return {
                    'action': 'PRESS',
                    'element_id': None,
                    'value': match.group(1),
                    'reasoning': response
                }
        
        # Fallback
        return {
            'action': 'WAIT',
            'element_id': None,
            'value': None,
            'reasoning': response
        }
    
    def _parse_mind2web_action(self, response: str) -> Dict[str, Any]:
        """Parse Mind2Web action format (CLICK/TYPE/SELECT commands)"""
        response = response.strip()
        parts = response.split()
        
        if not parts:
            return {'action': 'WAIT', 'element_id': None, 'value': None, 'reasoning': response}
        
        action_type = parts[0]
        
        if action_type == 'CLICK' and len(parts) >= 2:
            return {
                'action': 'CLICK',
                'element_id': parts[1],
                'value': None,
                'reasoning': response
            }
        elif action_type == 'TYPE' and len(parts) >= 3:
            return {
                'action': 'TYPE',
                'element_id': parts[1],
                'value': ' '.join(parts[2:]),
                'reasoning': response
            }
        elif action_type == 'SELECT' and len(parts) >= 3:
            return {
                'action': 'SELECT',
                'element_id': parts[1],
                'value': ' '.join(parts[2:]),
                'reasoning': response
            }
        
        # Fallback
        return {
            'action': 'WAIT',
            'element_id': None,
            'value': None,
            'reasoning': response
        }
