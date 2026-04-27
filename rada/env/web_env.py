"""
Web Environment - Unified Interface

Combines browser, DOM parser, and grounding into unified interface
"""

from typing import Dict, Any, List, Optional
from loguru import logger

from .browser_agent import BrowserAgent
from .dom_parser import DOMParser
from .grounding import ElementGrounding


class WebEnvironment:
    """Unified web environment"""
    
    def __init__(self, headless: bool = True):
        self.browser = BrowserAgent(headless=headless)
        self.dom_parser = DOMParser()
        self.grounding = ElementGrounding()
        self.elements = []
    
    async def start(self):
        """Start environment"""
        await self.browser.start()
    
    async def stop(self):
        """Stop environment"""
        await self.browser.stop()
    
    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate and observe"""
        html = await self.browser.navigate(url)
        return await self._observe()
    
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute grounded action"""
        action_type = action.get('action', '').upper()
        
        # For actions that don't need grounding, execute directly
        if action_type in ('NAVIGATE',):
            url = action.get('value', '')
            if url:
                await self.browser.navigate(url)
            return await self._observe()
        
        # For PRESS, SCROLL, WAIT - execute directly without grounding
        if action_type in ('PRESS', 'SCROLL', 'WAIT'):
            await self.browser.execute_action(action)
            return await self._observe()
        
        # Ground action to element
        grounded_action = self.grounding.ground_action(action, self.elements)
        
        if not grounded_action or not grounded_action.get('grounded'):
            logger.warning(f"Action not grounded: {action_type}")
            return await self._observe()
        
        # Execute in browser
        await self.browser.execute_action(grounded_action)
        
        return await self._observe()
    
    async def _observe(self) -> Dict[str, Any]:
        """Observe current state"""
        state = await self.browser.get_state()
        
        # Parse DOM
        self.elements = self.dom_parser.parse(state['html'])
        
        state['elements'] = self.elements
        state['formatted_elements'] = self.dom_parser.format_for_llm(self.elements)
        
        return state
