"""
Real Browser Agent with Playwright

Implements real browser automation with:
- Playwright-based browser control
- DOM parsing and element grounding
- Action execution with element IDs and XPath
- State tracking and observation
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from loguru import logger

from .advanced_dom_parser import AdvancedDOMParser


class RealBrowserAgent:
    """
    Real browser agent using Playwright
    
    Provides:
    - Full browser automation
    - DOM parsing with element grounding
    - Action execution with precise element targeting
    - State observation and tracking
    """
    
    def __init__(
        self,
        headless: bool = True,
        slow_mo: int = 50,
        viewport_width: int = 1280,
        viewport_height: int = 720
    ):
        """
        Initialize browser agent
        
        Args:
            headless: Run browser in headless mode
            slow_mo: Slow down operations (ms)
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
        """
        self.headless = headless
        self.slow_mo = slow_mo
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        self.dom_parser = AdvancedDOMParser()
        self.current_state: Dict[str, Any] = {}
        self.action_history: List[Dict[str, Any]] = []
        
        logger.info("RealBrowserAgent initialized")
    
    async def start(self):
        """Start the browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo
        )
        self.context = await self.browser.new_context(
            viewport={'width': self.viewport_width, 'height': self.viewport_height}
        )
        self.page = await self.context.new_page()
        
        logger.info("Browser started successfully")
    
    async def stop(self):
        """Stop the browser"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("Browser stopped")
    
    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> Dict[str, Any]:
        """
        Navigate to URL
        
        Args:
            url: URL to navigate to
            wait_until: When to consider navigation complete
            
        Returns:
            Page state after navigation
        """
        try:
            logger.info(f"Navigating to: {url}")
            await self.page.goto(url, wait_until=wait_until, timeout=60000)
            await asyncio.sleep(2)  # Additional wait for page to stabilize
            
            state = await self.observe()
            return state
            
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            raise
    
    async def click_element(self, element_id: str) -> Dict[str, Any]:
        """
        Click element by ID
        
        Args:
            element_id: Element ID from DOM parser
            
        Returns:
            Page state after click
        """
        try:
            element_info = self.dom_parser.get_element_by_id(element_id)
            if not element_info:
                raise ValueError(f"Element {element_id} not found")
            
            xpath = element_info['xpath']
            logger.info(f"Clicking element: {element_id} ({xpath})")
            
            # Try XPath first
            try:
                await self.page.click(f"xpath={xpath}", timeout=5000)
            except:
                # Fallback to CSS selector
                css = element_info['css_selector']
                await self.page.click(css, timeout=5000)
            
            await self._wait_for_page_load()
            
            # Record action
            self._record_action("CLICK", element_id, xpath)
            
            return await self.observe()
            
        except Exception as e:
            logger.error(f"Click failed: {e}")
            raise
    
    async def type_text(self, element_id: str, text: str, clear_first: bool = True) -> Dict[str, Any]:
        """
        Type text into element
        
        Args:
            element_id: Element ID
            text: Text to type
            clear_first: Clear existing text first
            
        Returns:
            Page state after typing
        """
        try:
            element_info = self.dom_parser.get_element_by_id(element_id)
            if not element_info:
                raise ValueError(f"Element {element_id} not found")
            
            xpath = element_info['xpath']
            logger.info(f"Typing into element: {element_id} - '{text}'")
            
            # Try XPath
            try:
                element = await self.page.query_selector(f"xpath={xpath}")
            except:
                # Fallback to CSS
                css = element_info['css_selector']
                element = await self.page.query_selector(css)
            
            if clear_first:
                await element.fill('')
            
            await element.type(text, delay=50)  # 50ms delay for realistic typing
            
            await self._wait_for_page_load()
            
            # Record action
            self._record_action("TYPE", element_id, xpath, {'text': text})
            
            return await self.observe()
            
        except Exception as e:
            logger.error(f"Type failed: {e}")
            raise
    
    async def select_option(self, element_id: str, value: str) -> Dict[str, Any]:
        """
        Select dropdown option
        
        Args:
            element_id: Element ID
            value: Option value to select
            
        Returns:
            Page state after selection
        """
        try:
            element_info = self.dom_parser.get_element_by_id(element_id)
            if not element_info:
                raise ValueError(f"Element {element_id} not found")
            
            xpath = element_info['xpath']
            logger.info(f"Selecting option: {element_id} - '{value}'")
            
            # Try XPath
            try:
                await self.page.select_option(f"xpath={xpath}", value=value)
            except:
                # Fallback to CSS
                css = element_info['css_selector']
                await self.page.select_option(css, value=value)
            
            await self._wait_for_page_load()
            
            # Record action
            self._record_action("SELECT", element_id, xpath, {'value': value})
            
            return await self.observe()
            
        except Exception as e:
            logger.error(f"Select failed: {e}")
            raise
    
    async def press_key(self, key: str) -> Dict[str, Any]:
        """
        Press keyboard key
        
        Args:
            key: Key to press (Enter, Tab, Escape, etc.)
            
        Returns:
            Page state after key press
        """
        try:
            logger.info(f"Pressing key: {key}")
            await self.page.keyboard.press(key)
            await self._wait_for_page_load()
            
            # Record action
            self._record_action("PRESS", None, None, {'key': key})
            
            return await self.observe()
            
        except Exception as e:
            logger.error(f"Key press failed: {e}")
            raise
    
    async def scroll(self, direction: str = "down", amount: int = 300) -> Dict[str, Any]:
        """
        Scroll page
        
        Args:
            direction: Direction (up/down)
            amount: Pixels to scroll
            
        Returns:
            Page state after scroll
        """
        try:
            logger.info(f"Scrolling {direction} by {amount}px")
            
            if direction == "down":
                await self.page.evaluate(f"window.scrollBy(0, {amount})")
            else:
                await self.page.evaluate(f"window.scrollBy(0, -{amount})")
            
            await asyncio.sleep(0.3)
            
            # Record action
            self._record_action("SCROLL", None, None, {'direction': direction, 'amount': amount})
            
            return await self.observe()
            
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            raise
    
    async def wait(self, milliseconds: int = 1000) -> Dict[str, Any]:
        """
        Wait for specified time
        
        Args:
            milliseconds: Time to wait
            
        Returns:
            Current page state
        """
        await asyncio.sleep(milliseconds / 1000)
        
        # Record action
        self._record_action("WAIT", None, None, {'milliseconds': milliseconds})
        
        return await self.observe()
    
    async def execute_action(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action from action dictionary
        
        Args:
            action_dict: Action dictionary with action type and parameters
            
        Returns:
            Page state after action
        """
        action_type = action_dict.get('action', '').upper()
        element_id = action_dict.get('element_id')
        value = action_dict.get('value')
        
        action_map = {
            'CLICK': lambda: self.click_element(element_id),
            'TYPE': lambda: self.type_text(element_id, value),
            'SELECT': lambda: self.select_option(element_id, value),
            'PRESS': lambda: self.press_key(value),
            'SCROLL': lambda: self.scroll(value if value else "down"),
            'WAIT': lambda: self.wait(int(value) if value else 1000),
        }
        
        if action_type not in action_map:
            raise ValueError(f"Unknown action type: {action_type}")
        
        return await action_map[action_type]()
    
    async def observe(self) -> Dict[str, Any]:
        """
        Observe current page state
        
        Returns:
            Complete page state observation
        """
        try:
            # Get basic page info
            url = self.page.url
            title = await self.page.title()
            html = await self.page.content()
            
            # Parse DOM with error handling
            try:
                dom_result = self.dom_parser.parse_and_index(html)
            except Exception as e:
                logger.error(f"DOM parsing failed: {e}")
                dom_result = {'elements': [], 'element_index': {}}
            
            if dom_result is None:
                dom_result = {'elements': [], 'element_index': {}}
            elements = dom_result.get('elements', [])
            if elements is None:
                elements = []
            element_index = dom_result.get('element_index', {})
            if element_index is None:
                element_index = {}
            
            # Create accessibility tree with error handling
            try:
                accessibility_tree = self.dom_parser.create_accessibility_tree(html, max_depth=2)
            except Exception as e:
                logger.error(f"Accessibility tree creation failed: {e}")
                accessibility_tree = ""
            
            if accessibility_tree is None:
                accessibility_tree = ""
            
            # Format elements for LLM with error handling
            try:
                formatted_elements = self.dom_parser.format_elements_for_llm(elements, max_elements=30)
            except Exception as e:
                logger.error(f"Element formatting failed: {e}")
                formatted_elements = ""
            
            if formatted_elements is None:
                formatted_elements = ""
            
            # Build state
            self.current_state = {
                'url': url,
                'title': title,
                'html': html,
                'elements': elements,
                'element_index': element_index,
                'formatted_elements': formatted_elements,
                'accessibility_tree': accessibility_tree,
                'num_elements': len(elements),
                'timestamp': time.time()
            }
            
            logger.debug(f"Observed page: {title} ({len(elements)} elements)")
            
            return self.current_state
            
        except Exception as e:
            logger.error(f"Observation failed: {e}")
            # Return minimal state on error
            self.current_state = {
                'url': self.page.url if self.page else '',
                'title': '',
                'html': '',
                'elements': [],
                'element_index': {},
                'formatted_elements': '',
                'accessibility_tree': '',
                'num_elements': 0,
                'timestamp': time.time()
            }
            return self.current_state
    
    async def _wait_for_page_load(self, timeout: int = 5000):
        """
        Wait for page to stabilize after action
        
        Args:
            timeout: Maximum wait time in ms
        """
        try:
            await asyncio.sleep(0.5)  # Initial wait
            await self.page.wait_for_load_state('networkidle', timeout=timeout)
            await asyncio.sleep(0.3)  # Additional wait for JS rendering
        except:
            # If wait fails, continue anyway
            await asyncio.sleep(0.5)
    
    def _record_action(self, action_type: str, element_id: Optional[str], 
                      xpath: Optional[str], metadata: Optional[Dict] = None):
        """
        Record action in history
        
        Args:
            action_type: Type of action
            element_id: Element ID
            xpath: Element XPath
            metadata: Additional metadata
        """
        action_record = {
            'action': action_type,
            'element_id': element_id,
            'xpath': xpath,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        self.action_history.append(action_record)
        logger.debug(f"Action recorded: {action_type} on {element_id}")
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get action history"""
        return self.action_history.copy()
    
    def clear_history(self):
        """Clear action history"""
        self.action_history = []
    
    async def take_screenshot(self, path: str = "screenshot.png"):
        """
        Take screenshot
        
        Args:
            path: File path to save screenshot
        """
        await self.page.screenshot(path=path, full_page=True)
        logger.info(f"Screenshot saved to {path}")
