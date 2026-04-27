"""
Advanced DOM Parser with XPath Grounding and Element IDs

Provides comprehensive DOM parsing, element identification, and XPath generation
for web agent action grounding as described in the RaDA paper.
"""

import hashlib
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup, Tag
from loguru import logger


class AdvancedDOMParser:
    """
    Advanced DOM parser with XPath grounding and unique element identification
    
    Supports:
    - Unique element ID generation
    - XPath generation for precise element location
    - Interactive element extraction with full attributes
    - Accessibility tree simplification
    - Action grounding for web agents
    """
    
    def __init__(self):
        """Initialize the advanced DOM parser"""
        self.element_map: Dict[str, Dict] = {}
    
    def parse_and_index(self, html: str) -> Dict[str, Any]:
        """
        Parse HTML and create indexed element map
        
        Args:
            html: HTML content
            
        Returns:
            Dictionary with parsed DOM information
        """
        soup = BeautifulSoup(html, 'lxml')
        self.element_map = {}
        
        # Extract all interactive elements
        interactive_elements = self._extract_interactive_elements(soup)
        
        # Generate element map
        element_index = {}
        for idx, elem_info in enumerate(interactive_elements):
            element_id = f"elem_{idx:04d}"
            elem_info['element_id'] = element_id
            element_index[element_id] = elem_info
        
        self.element_map = element_index
        
        return {
            'soup': soup,
            'elements': interactive_elements,
            'element_index': element_index,
            'total_elements': len(interactive_elements)
        }
    
    def _extract_interactive_elements(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract all interactive elements from DOM
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of interactive element information
        """
        elements = []
        
        # Define interactive tags
        interactive_tags = {
            'a': ['href'],
            'button': [],
            'input': ['type', 'placeholder', 'value'],
            'select': [],
            'textarea': ['placeholder'],
            '[role="button"]': [],
            '[role="link"]': [],
            '[role="textbox"]': [],
            '[contenteditable="true"]': []
        }
        
        # Extract elements
        all_elements = []
        
        # Standard interactive tags
        for tag_name in ['a', 'button', 'input', 'select', 'textarea']:
            for elem in soup.find_all(tag_name):
                if self._is_visible_element(elem):
                    all_elements.append(elem)
        
        # Role-based interactive elements
        try:
            for elem in soup.find_all(attrs={'role': lambda x: x and ('button' in x or 'link' in x)}):
                if elem and self._is_visible_element(elem):
                    all_elements.append(elem)
        except Exception as e:
            logger.warning(f"Failed to extract role-based elements: {e}")
        
        # Process each element
        for idx, elem in enumerate(all_elements):
            elem_info = self._process_element(elem, idx)
            if elem_info:
                elements.append(elem_info)
        
        logger.debug(f"Extracted {len(elements)} interactive elements")
        return elements
    
    def _process_element(self, elem: Tag, index: int) -> Optional[Dict[str, Any]]:
        """
        Process a single DOM element
        
        Args:
            elem: BeautifulSoup Tag
            index: Element index
            
        Returns:
            Element information dictionary
        """
        try:
            tag_name = elem.name
            text = self._get_element_text(elem)
            xpath = self._generate_xpath(elem)
            css_selector = self._generate_css_selector(elem)
            
            # Generate unique element ID
            elem_id = self._generate_element_id(elem, index)
            
            # Extract attributes
            attributes = {
                'tag': tag_name,
                'id': elem.get('id', ''),
                'name': elem.get('name', ''),
                'type': elem.get('type', ''),
                'class': elem.get('class', []),
                'placeholder': elem.get('placeholder', ''),
                'value': elem.get('value', ''),
                'href': elem.get('href', ''),
                'role': elem.get('role', ''),
                'aria_label': elem.get('aria-label', ''),
                'disabled': elem.has_attr('disabled'),
                'readonly': elem.has_attr('readonly'),
            }
            
            # Determine element type and possible actions
            action_type = self._determine_action_type(elem)
            
            elem_info = {
                'element_id': elem_id,
                'tag': tag_name,
                'text': text[:200],  # Limit text length
                'xpath': xpath,
                'css_selector': css_selector,
                'attributes': attributes,
                'action_type': action_type,
                'is_interactive': True,
                'index': index
            }
            
            return elem_info
            
        except Exception as e:
            logger.warning(f"Failed to process element: {e}")
            return None
    
    def _generate_element_id(self, elem: Tag, index: int) -> str:
        """
        Generate unique element ID based on content and position
        
        Args:
            elem: BeautifulSoup Tag
            index: Element index
            
        Returns:
            Unique element ID
        """
        # Use element position and hash of key attributes
        elem_hash = hashlib.md5(
            f"{elem.name}_{elem.get('id', '')}_{elem.get('name', '')}_{index}".encode()
        ).hexdigest()[:8]
        
        return f"elem_{index:04d}_{elem_hash}"
    
    def _generate_xpath(self, elem: Tag) -> str:
        """
        Generate XPath for element
        
        Args:
            elem: BeautifulSoup Tag
            
        Returns:
            XPath string
        """
        parts = []
        current = elem
        
        while current and current.name != '[document]':
            tag = current.name
            
            # Get sibling index
            siblings = [s for s in current.parent.children if s.name == tag] if current.parent else []
            if len(siblings) > 1:
                idx = siblings.index(current) + 1
                parts.append(f"{tag}[{idx}]")
            else:
                parts.append(tag)
            
            current = current.parent
        
        return '/' + '/'.join(reversed(parts))
    
    def _generate_css_selector(self, elem: Tag) -> str:
        """
        Generate CSS selector for element
        
        Args:
            elem: BeautifulSoup Tag
            
        Returns:
            CSS selector string
        """
        parts = []
        current = elem
        
        while current and current.name != '[document]':
            selector = current.name
            
            if current.get('id'):
                selector += f"#{current['id']}"
                parts.append(selector)
                break
            elif current.get('class'):
                classes = '.'.join(current['class'][:2])  # Limit to 2 classes
                selector += f".{classes}"
            
            parts.append(selector)
            current = current.parent
        
        return ' > '.join(reversed(parts))
    
    def _determine_action_type(self, elem: Tag) -> str:
        """
        Determine the primary action type for an element
        
        Args:
            elem: BeautifulSoup Tag
            
        Returns:
            Action type string
        """
        tag = elem.name.lower()
        
        if tag == 'a' or elem.get('role') == 'link':
            return 'CLICK'
        elif tag == 'button' or elem.get('role') == 'button':
            return 'CLICK'
        elif tag == 'input':
            input_type = elem.get('type', 'text').lower()
            if input_type in ['text', 'email', 'password', 'search', 'tel', 'url']:
                return 'TYPE'
            elif input_type in ['submit', 'button', 'reset']:
                return 'CLICK'
            elif input_type == 'checkbox':
                return 'CLICK'
            elif input_type == 'radio':
                return 'CLICK'
        elif tag == 'select':
            return 'SELECT'
        elif tag == 'textarea':
            return 'TYPE'
        elif elem.get('contenteditable') == 'true':
            return 'TYPE'
        
        return 'CLICK'  # Default
    
    def _get_element_text(self, elem: Tag) -> str:
        """
        Get text content of element
        
        Args:
            elem: BeautifulSoup Tag
            
        Returns:
            Text content
        """
        # Get direct text or aria-label
        text = elem.get('aria-label', '') or elem.get('title', '')
        
        if not text:
            # Get text content, excluding child tags
            text = elem.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _is_visible_element(self, elem: Tag) -> bool:
        """
        Check if element is likely visible
        
        Args:
            elem: BeautifulSoup Tag
            
        Returns:
            True if element is visible
        """
        # Skip hidden inputs
        if elem.name == 'input' and elem.get('type', '').lower() == 'hidden':
            return False
        
        # Skip elements with display:none or visibility:hidden
        style = elem.get('style', '').lower()
        if 'display:none' in style or 'visibility:hidden' in style:
            return False
        
        # Skip empty elements
        if not elem.get_text(strip=True) and not elem.get('aria-label'):
            if elem.name not in ['input', 'img', 'button']:
                return False
        
        return True
    
    def format_elements_for_llm(self, elements: List[Dict[str, Any]], max_elements: int = 50) -> str:
        """
        Format elements for LLM prompt
        
        Args:
            elements: List of element information
            max_elements: Maximum number of elements to include
            
        Returns:
            Formatted string
        """
        if not elements:
            return "No interactive elements found on the page."
        
        # Limit to max_elements
        elements = elements[:max_elements]
        
        formatted = []
        formatted.append(f"Found {len(elements)} interactive elements:\n")
        
        for elem in elements:
            elem_id = elem['element_id']
            tag = elem['tag']
            text = elem['text']
            action_type = elem['action_type']
            xpath = elem['xpath']
            
            # Format element info
            line = f"[{elem_id}] <{tag}>"
            
            if text:
                line += f' text="{text[:50]}"'
            
            line += f' action={action_type}'
            
            # Add important attributes
            attrs = elem['attributes']
            if attrs.get('placeholder'):
                line += f' placeholder="{attrs["placeholder"]}"'
            if attrs.get('type'):
                line += f' type="{attrs["type"]}"'
            
            formatted.append(line)
        
        return '\n'.join(formatted)
    
    def get_element_by_id(self, element_id: str) -> Optional[Dict[str, Any]]:
        """
        Get element information by ID
        
        Args:
            element_id: Element ID
            
        Returns:
            Element information or None
        """
        return self.element_map.get(element_id)
    
    def get_element_by_xpath(self, xpath: str, html: str) -> Optional[Tag]:
        """
        Find element by XPath
        
        Args:
            xpath: XPath string
            html: HTML content
            
        Returns:
            BeautifulSoup Tag or None
        """
        try:
            soup = BeautifulSoup(html, 'lxml')
            # Use lxml's xpath support
            from lxml import etree
            
            # Parse HTML with lxml
            tree = etree.HTML(str(soup))
            elements = tree.xpath(xpath)
            
            if elements:
                return elements[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to find element by XPath: {e}")
            return None
    
    def create_accessibility_tree(self, html: str, max_depth: int = 3) -> str:
        """
        Create simplified accessibility tree
        
        Args:
            html: HTML content
            max_depth: Maximum tree depth
            
        Returns:
            Accessibility tree string
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove non-essential elements
        for tag in soup(['script', 'style', 'meta', 'link', 'head']):
            tag.decompose()
        
        # Build tree
        tree_lines = []
        self._build_tree_recursive(soup.body or soup, tree_lines, 0, max_depth)
        
        return '\n'.join(tree_lines)
    
    def _build_tree_recursive(self, elem: Tag, lines: List[str], depth: int, max_depth: int):
        """
        Recursively build accessibility tree
        
        Args:
            elem: Current element
            lines: Output lines
            depth: Current depth
            max_depth: Maximum depth
        """
        if depth > max_depth:
            return
        
        # Get element info
        tag = elem.name
        text = elem.get('aria-label', '') or elem.get_text(strip=True)[:50]
        
        if text:
            indent = '  ' * depth
            lines.append(f"{indent}[{tag}] {text}")
        
        # Process children
        for child in elem.children:
            if isinstance(child, Tag):
                self._build_tree_recursive(child, lines, depth + 1, max_depth)
