"""
DOM Parser with pruning

Parses HTML and extracts interactive elements with DOM pruning
"""

from typing import List, Dict, Any
from bs4 import BeautifulSoup
from loguru import logger


class DOMParser:
    """DOM parser with pruning for web agents"""
    
    def parse(self, html: str) -> List[Dict[str, Any]]:
        """Parse HTML and extract interactive elements"""
        soup = BeautifulSoup(html, 'lxml')
        
        # Prune non-essential elements
        self._prune_dom(soup)
        
        # Extract interactive elements
        elements = []
        for idx, elem in enumerate(self._get_interactive_elements(soup)):
            elements.append({
                'element_id': f'elem_{idx:04d}',
                'tag': elem.name,
                'text': self._get_text(elem)[:100],
                'xpath': self._get_xpath(elem),
                'attributes': self._get_attrs(elem),
                'action_type': self._infer_action(elem)
            })
        
        return elements
    
    def _prune_dom(self, soup: BeautifulSoup):
        """Remove non-essential elements"""
        for tag in soup(['script', 'style', 'meta', 'link', 'head', 'noscript']):
            tag.decompose()
    
    def _get_interactive_elements(self, soup: BeautifulSoup):
        """Get interactive elements"""
        elements = []
        for tag in ['a', 'button', 'input', 'select', 'textarea']:
            elements.extend(soup.find_all(tag))
        return elements
    
    def _get_text(self, elem) -> str:
        """Get element text"""
        return elem.get('aria-label', '') or elem.get_text(strip=True)
    
    def _get_xpath(self, elem) -> str:
        """Generate a more robust XPath using id, name, or indexed path"""
        # Prefer CSS selector with id if available
        elem_id = elem.get('id', '')
        if elem_id:
            return f'//*[@id="{elem_id}"]'
        
        # Prefer name attribute if available
        elem_name = elem.get('name', '')
        if elem_name:
            return f'//{elem.name}[@name="{elem_name}"]'
        
        # Build indexed path for uniqueness
        parts = []
        current = elem
        while current and current.name != '[document]':
            tag = current.name
            # Count siblings with same tag for indexing
            parent = current.parent
            if parent:
                siblings = [c for c in parent.children if hasattr(c, 'name') and c.name == tag]
                if len(siblings) > 1:
                    idx = siblings.index(current) + 1
                    parts.append(f'{tag}[{idx}]')
                else:
                    parts.append(tag)
            else:
                parts.append(tag)
            current = parent
        
        return '/' + '/'.join(reversed(parts))
    
    def _get_attrs(self, elem) -> Dict:
        """Get important attributes"""
        return {
            'id': elem.get('id', ''),
            'name': elem.get('name', ''),
            'type': elem.get('type', ''),
            'class': elem.get('class', [])
        }
    
    def _infer_action(self, elem) -> str:
        """Infer action type"""
        tag = elem.name
        if tag in ['a', 'button']:
            return 'CLICK'
        elif tag == 'input':
            input_type = elem.get('type', 'text')
            if input_type in ['text', 'email', 'password']:
                return 'TYPE'
            return 'CLICK'
        elif tag == 'select':
            return 'SELECT'
        return 'CLICK'
    
    def format_for_llm(self, elements: List[Dict], max_elements: int = 30) -> str:
        """Format elements for LLM"""
        if not elements:
            return "No interactive elements found"
        
        formatted = []
        for elem in elements[:max_elements]:
            line = f"[{elem['element_id']}] <{elem['tag']}>"
            attrs = elem.get('attributes', {})
            # Add useful attributes
            if attrs.get('id'):
                line += f' id="{attrs["id"]}"'
            if attrs.get('name'):
                line += f' name="{attrs["name"]}"'
            if attrs.get('type'):
                line += f' type="{attrs["type"]}"'
            if elem['text']:
                line += f' text="{elem["text"][:50]}"'
            line += f' action={elem["action_type"]}'
            formatted.append(line)
        
        return '\n'.join(formatted)
