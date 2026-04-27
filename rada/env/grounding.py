"""
Element Grounding

Maps LLM actions to DOM elements using text-based matching
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger


class ElementGrounding:
    """Ground LLM actions to DOM elements"""
    
    def __init__(self):
        self.element_embeddings = {}
    
    def ground_action(
        self,
        action: Dict[str, Any],
        elements: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Ground action to specific element
        
        Args:
            action: Action from LLM
            elements: Available elements
            
        Returns:
            Grounded action with element XPath
        """
        action_type = action.get('action', '').upper()
        
        # Actions that don't need element grounding
        no_element_actions = {'WAIT', 'NAVIGATE', 'PRESS', 'SCROLL'}
        if action_type in no_element_actions:
            action['grounded'] = True
            return action
        
        element_id = action.get('element_id')
        
        # Try exact element_id match first
        if element_id:
            for elem in elements:
                if elem['element_id'] == element_id:
                    action['xpath'] = elem['xpath']
                    action['grounded'] = True
                    return action
        
        # Fallback 1: match by element text / description
        grounded = self._match_by_text_and_action(action, elements)
        if grounded:
            return grounded
        
        # Fallback 2: match by action type to first suitable element
        grounded = self._match_by_action_type(action, elements)
        if grounded:
            return grounded
        
        logger.warning(f"Could not ground action: {action_type} element_id={element_id}")
        return None
    
    def _match_by_text_and_action(
        self,
        action: Dict[str, Any],
        elements: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Match action to element using text from value, element_id, or reasoning"""
        action_type = action.get('action', '').upper()
        
        # Gather candidate search terms from the action
        search_terms = []
        value = action.get('value') or ''
        if value:
            search_terms.append(str(value).lower())
        
        element_id = action.get('element_id') or ''
        if element_id and not element_id.startswith('elem_'):
            search_terms.append(str(element_id).lower())
        
        # Extract keywords from reasoning
        reasoning = action.get('reasoning', '') or ''
        reasoning_lower = reasoning.lower()
        for keyword in ['search', 'button', 'link', 'input', 'menu', 'login', 'submit', 'click', 'type', 'enter']:
            if keyword in reasoning_lower:
                search_terms.append(keyword)
        
        if not search_terms:
            return None
        
        best_match = None
        best_score = 0
        
        for elem in elements:
            # Check element text
            elem_text = (elem.get('text', '') or '').lower()
            # Check element attributes
            attrs = elem.get('attributes', {})
            attr_text = ' '.join([
                str(attrs.get('id', '')),
                str(attrs.get('name', '')),
                str(attrs.get('type', ''))
            ]).lower()
            
            # Also check tag name for action type compatibility
            tag = elem.get('tag', '')
            
            score = 0
            for term in search_terms:
                if term in elem_text:
                    score += 2
                if term in attr_text:
                    score += 1
            
            
            # Bonus for action-type compatibility
            if action_type == 'CLICK' and tag in ('a', 'button'):
                score += 1
            elif action_type == 'TYPE' and tag in ('input', 'textarea'):
                score += 2
            elif action_type == 'SELECT' and tag == 'select':
                score += 2
            
            if score > best_score:
                best_score = score
                best_match = elem
        
        if best_match and best_score >= 2:
            action['xpath'] = best_match['xpath']
            action['grounded'] = True
            action['match_score'] = best_score
            action['matched_element_id'] = best_match['element_id']
            return action
        
        return None
    
    def _match_by_action_type(
        self,
        action: Dict[str, Any],
        elements: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Match action to first suitable element by action type"""
        action_type = action.get('action', '').upper()
        
        # Find first element matching action type
        for elem in elements:
            tag = elem.get('tag', '')
            inferred_action = elem.get('action_type', '')
            
            if action_type == 'CLICK' and tag in ('a', 'button'):
                action['xpath'] = elem['xpath']
                action['grounded'] = True
                action['matched_element_id'] = elem['element_id']
                return action
            elif action_type == 'TYPE' and tag in ('input', 'textarea'):
                action['xpath'] = elem['xpath']
                action['grounded'] = True
                action['matched_element_id'] = elem['element_id']
                return action
            elif action_type == 'SELECT' and tag == 'select':
                action['xpath'] = elem['xpath']
                action['grounded'] = True
                action['matched_element_id'] = elem['element_id']
                return action
            
            # Also match by inferred action type
            if inferred_action == action_type:
                action['xpath'] = elem['xpath']
                action['grounded'] = True
                action['matched_element_id'] = elem['element_id']
                return action
        
        
        return None
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        return overlap / max(len(words1), len(words2))
