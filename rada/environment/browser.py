"""
Browser automation using Playwright
"""

import asyncio
from typing import Optional, Dict, Any, List
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from loguru import logger


class BrowserEnvironment:
    """Manages browser automation for web agent"""
    
    def __init__(self, headless: bool = False, slow_mo: int = 100):
        """
        Initialize browser environment
        
        Args:
            headless: Whether to run browser in headless mode
            slow_mo: Slow down operations by specified milliseconds (for debugging)
        """
        self.headless = headless
        self.slow_mo = slow_mo
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
    
    async def start(self):
        """Start the browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo
        )
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        logger.info("Browser started")
    
    async def stop(self):
        """Stop the browser"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("Browser stopped")
    
    async def navigate(self, url: str) -> Dict[str, Any]:
        """
        Navigate to a URL
        
        Args:
            url: URL to navigate to
            
        Returns:
            Page state information
        """
        try:
            response = await self.page.goto(url, wait_until="networkidle")
            logger.info(f"Navigated to {url}")
            
            return await self.get_page_state()
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            raise
    
    async def click(self, selector: str) -> Dict[str, Any]:
        """
        Click an element
        
        Args:
            selector: CSS selector for the element
            
        Returns:
            Page state after click
        """
        try:
            # Try normal click first
            await self.page.click(selector, timeout=5000)
        except Exception as e:
            # If normal click fails, try force click
            logger.warning(f"Normal click failed, trying force click: {e}")
            try:
                await self.page.click(selector, force=True, timeout=5000)
            except Exception as e2:
                logger.error(f"Force click also failed: {e2}")
                raise
        
        await asyncio.sleep(0.5)  # Brief wait after click
        logger.info(f"Clicked element: {selector}")
        return await self.get_page_state()
    
    async def type_text(self, selector: str, text: str) -> Dict[str, Any]:
        """
        Type text into an input field
        
        Args:
            selector: CSS selector for the input
            text: Text to type
            
        Returns:
            Page state after typing
        """
        try:
            await self.page.fill(selector, text)
            logger.info(f"Typed text into: {selector}")
            return await self.get_page_state()
        except Exception as e:
            logger.error(f"Type failed: {e}")
            raise
    
    async def press_key(self, key: str) -> Dict[str, Any]:
        """
        Press a keyboard key
        
        Args:
            key: Key to press (e.g., "Enter", "Tab")
            
        Returns:
            Page state after key press
        """
        try:
            await self.page.press("body", key)
            await self.page.wait_for_load_state("networkidle")
            logger.info(f"Pressed key: {key}")
            return await self.get_page_state()
        except Exception as e:
            logger.error(f"Key press failed: {e}")
            raise
    
    async def select_option(self, selector: str, value: str) -> Dict[str, Any]:
        """
        Select an option from a dropdown
        
        Args:
            selector: CSS selector for the select element
            value: Value to select
            
        Returns:
            Page state after selection
        """
        try:
            await self.page.select_option(selector, value)
            logger.info(f"Selected option: {value} in {selector}")
            return await self.get_page_state()
        except Exception as e:
            logger.error(f"Select failed: {e}")
            raise
    
    async def scroll(self, direction: str = "down", amount: int = 300) -> Dict[str, Any]:
        """
        Scroll the page
        
        Args:
            direction: Direction to scroll ("up" or "down")
            amount: Amount to scroll in pixels
            
        Returns:
            Page state after scroll
        """
        try:
            if direction == "down":
                await self.page.evaluate(f"window.scrollBy(0, {amount})")
            else:
                await self.page.evaluate(f"window.scrollBy(0, -{amount})")
            logger.info(f"Scrolled {direction} by {amount}px")
            return await self.get_page_state()
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            raise
    
    async def wait(self, milliseconds: int = 1000) -> Dict[str, Any]:
        """
        Wait for specified time
        
        Args:
            milliseconds: Time to wait in milliseconds
            
        Returns:
            Current page state
        """
        await asyncio.sleep(milliseconds / 1000)
        logger.info(f"Waited for {milliseconds}ms")
        return await self.get_page_state()
    
    async def get_page_state(self) -> Dict[str, Any]:
        """
        Get current page state
        
        Returns:
            Dictionary containing page state information
        """
        try:
            title = await self.page.title()
            url = self.page.url
            
            # Get page content summary
            content = await self.page.content()
            
            return {
                "title": title,
                "url": url,
                "content": content,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Failed to get page state: {e}")
            return {
                "title": "",
                "url": "",
                "content": "",
                "status": "error"
            }
    
    async def execute_action(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a high-level action
        
        Args:
            action: Action type (CLICK, TYPE, SELECT, SCROLL, NAVIGATE, WAIT, PRESS)
            **kwargs: Action-specific parameters
            
        Returns:
            Page state after action
        """
        actions = {
            "CLICK": lambda: self.click(kwargs.get("selector")),
            "TYPE": lambda: self.type_text(kwargs.get("selector"), kwargs.get("text", "")),
            "SELECT": lambda: self.select_option(kwargs.get("selector"), kwargs.get("value", "")),
            "SCROLL": lambda: self.scroll(kwargs.get("direction", "down"), kwargs.get("amount", 300)),
            "NAVIGATE": lambda: self.navigate(kwargs.get("url", "")),
            "WAIT": lambda: self.wait(kwargs.get("milliseconds", 1000)),
            "PRESS": lambda: self.press_key(kwargs.get("key", "Enter")),
        }
        
        if action not in actions:
            raise ValueError(f"Unknown action: {action}")
        
        return await actions[action]()
