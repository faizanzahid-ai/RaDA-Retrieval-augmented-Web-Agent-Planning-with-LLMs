"""
DOM parsing and element extraction
"""

from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from loguru import logger


class DOMParser:
    """Parses and extracts information from HTML DOM"""
    
    def __init__(self):
        """Initialize the DOM parser"""
        pass
    
    def parse_html(self, html: str) -> BeautifulSoup:
        """
        Parse HTML content
        
        Args:
            html: HTML string
            
        Returns:
            BeautifulSoup object
        """
        soup = BeautifulSoup(html, 'lxml')
        return soup
    
    def extract_interactive_elements(self, html: str) -> List[Dict[str, Any]]:
        """
        Extract interactive elements from HTML
        
        Args:
            html: HTML string
            
        Returns:
            List of interactive elements with their attributes
        """
        soup = self.parse_html(html)
        elements = []
        
        # Extract clickable elements
        for i, elem in enumerate(soup.find_all(['a', 'button', 'input', 'select', 'textarea'])):
            element_info = {
                "id": elem.get('id', f"elem_{i}"),
                "tag": elem.name,
                "type": elem.get('type', ''),
                "text": elem.get_text(strip=True)[:100],
                "name": elem.get('name', ''),
                "placeholder": elem.get('placeholder', ''),
                "href": elem.get('href', ''),
                "css_selector": self._get_css_selector(elem),
            }
            elements.append(element_info)
        
        logger.info(f"Extracted {len(elements)} interactive elements")
        return elements
    
    def extract_page_summary(self, html: str) -> str:
        """
        Extract a summary of the page content
        
        Args:
            html: HTML string
            
        Returns:
            Page summary text
        """
        soup = self.parse_html(html)
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Limit to first 1000 characters
        summary = text[:1000]
        
        return summary
    
    def get_element_by_selector(self, html: str, selector: str) -> Optional[Dict[str, Any]]:
        """
        Get element by CSS selector
        
        Args:
            html: HTML string
            selector: CSS selector
            
        Returns:
            Element information or None
        """
        soup = self.parse_html(html)
        elem = soup.select_one(selector)
        
        if elem is None:
            return None
        
        return {
            "tag": elem.name,
            "text": elem.get_text(strip=True),
            "attributes": dict(elem.attrs),
        }
    
    def format_elements_for_prompt(self, elements: List[Dict[str, Any]]) -> str:
        """
        Format interactive elements for LLM prompt
        
        Args:
            elements: List of interactive elements
            
        Returns:
            Formatted string
        """
        if not elements:
            return "No interactive elements found."
        
        formatted = []
        for elem in elements[:20]:  # Limit to first 20 elements
            formatted.append(
                f"- [{elem['id']}] <{elem['tag']}> "
                f"text: '{elem['text']}' "
                f"type: '{elem['type']}'"
            )
        
        return "\n".join(formatted)
    
    def _get_css_selector(self, elem) -> str:
        """
        Generate CSS selector for an element
        
        Args:
            elem: BeautifulSoup element
            
        Returns:
            CSS selector string
        """
        if elem.get('id'):
            return f"#{elem['id']}"
        
        path = []
        while elem and elem.name != '[document]':
            selector = elem.name
            if elem.get('class'):
                selector += '.' + '.'.join(elem.get('class'))
            path.append(selector)
            elem = elem.parent
        
        return ' > '.join(reversed(path))
    
    def extract_form_fields(self, html: str) -> List[Dict[str, Any]]:
        """
        Extract form fields from HTML
        
        Args:
            html: HTML string
            
        Returns:
            List of form field information
        """
        soup = self.parse_html(html)
        fields = []
        
        for form in soup.find_all('form'):
            for field in form.find_all(['input', 'select', 'textarea']):
                field_info = {
                    "name": field.get('name', ''),
                    "type": field.get('type', 'text'),
                    "placeholder": field.get('placeholder', ''),
                    "required": field.has_attr('required'),
                    "css_selector": self._get_css_selector(field),
                }
                fields.append(field_info)
        
        return fields
