"""
Browser Agent with Playwright

Real browser automation with action execution
"""

import asyncio
from typing import Dict, Any, Optional, List
from playwright.async_api import async_playwright, Page, Browser
from loguru import logger


class BrowserAgent:
    """Playwright-based browser agent"""
    
    def __init__(self, headless: bool = True, slow_mo: int = 50):
        self.headless = headless
        self.slow_mo = slow_mo
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._action_history = []
    
    async def start(self):
        """Start browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo
        )
        self.page = await self.browser.new_page()
        logger.info("Browser started")
    
    async def stop(self):
        """Stop browser"""
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        logger.info("Browser stopped")
    
    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to URL and return page state"""
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            logger.warning(f"Navigation with domcontentloaded failed, trying load: {e}")
            try:
                await self.page.goto(url, wait_until="load", timeout=60000)
            except Exception as e2:
                logger.error(f"Navigation failed: {e2}")
                # Still try to get content if page partially loaded
        return await self.get_state()
    
    async def click(self, xpath: str):
        """Click element by XPath"""
        try:
            # First try to scroll the element into view
            try:
                locator = self.page.locator(f"xpath={xpath}").first
                await locator.scroll_into_view_if_needed(timeout=5000)
            except Exception:
                pass  # Scroll might fail, try clicking anyway
            await self.page.click(f"xpath={xpath}", timeout=10000)
        except Exception as e:
            logger.warning(f"Normal click failed, trying force click: {e}")
            try:
                # Scroll into view before force click too
                try:
                    locator = self.page.locator(f"xpath={xpath}").first
                    await locator.scroll_into_view_if_needed(timeout=3000)
                except Exception:
                    pass
                await self.page.click(f"xpath={xpath}", force=True, timeout=5000)
            except Exception as e2:
                # Last resort: use JavaScript click
                logger.warning(f"Force click failed, trying JS click: {e2}")
                try:
                    await self.page.evaluate(f"""() => {{
                        const result = document.evaluate('{xpath}', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                        const el = result.singleNodeValue;
                        if (el) {{ el.scrollIntoView({{block: 'center'}}); el.click(); }}
                    }}""")
                except Exception as e3:
                    logger.error(f"All click methods failed: {e3}")
                    raise
        await self._wait_stable()
    
    async def type_text(self, xpath: str, text: str):
        """Type text into element"""
        # Scroll element into view first
        try:
            locator = self.page.locator(f"xpath={xpath}").first
            await locator.scroll_into_view_if_needed(timeout=5000)
        except Exception:
            pass
        try:
            await self.page.fill(f"xpath={xpath}", '', timeout=10000)
        except Exception:
            try:
                await self.page.fill(f"xpath={xpath}", '', force=True, timeout=5000)
            except Exception:
                pass  # Clear might fail, try typing anyway
        await self.page.type(f"xpath={xpath}", text, delay=50, timeout=10000)
        await self._wait_stable()
    
    async def select_option(self, xpath: str, value: str):
        """Select dropdown option"""
        await self.page.select_option(f"xpath={xpath}", value=value)
        await self._wait_stable()
    
    async def press_key(self, key: str):
        """Press keyboard key"""
        await self.page.keyboard.press(key)
        await self._wait_stable()
    
    async def scroll(self, direction: str = "down", amount: int = 300):
        """Scroll page"""
        if direction == "down":
            await self.page.evaluate(f"window.scrollBy(0, {amount})")
        else:
            await self.page.evaluate(f"window.scrollBy(0, -{amount})")
        await asyncio.sleep(0.3)
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current page state"""
        return {
            'url': self.page.url,
            'title': await self.page.title(),
            'html': await self.page.content()
        }
    
    async def observe(self) -> Dict[str, Any]:
        """Observe current page state (alias for get_state with accessibility info)"""
        state = await self.get_state()
        # Add accessibility tree approximation from HTML
        state['accessibility_tree'] = state.get('html', '')[:3000]
        return state
    
    async def take_screenshot(self, path: str = None) -> Optional[bytes]:
        """Take a screenshot of the current page"""
        try:
            if path:
                from pathlib import Path
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                await self.page.screenshot(path=path)
                logger.info(f"Screenshot saved to {path}")
                return None
            else:
                return await self.page.screenshot()
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get the history of actions executed"""
        return self._action_history.copy()
    
    async def execute_action(self, action: Dict[str, Any]):
        """Execute action dictionary"""
        action_type = action.get('action', '').upper()
        xpath = action.get('xpath')
        value = action.get('value')
        
        try:
            if action_type == 'CLICK' and xpath:
                await self.click(xpath)
            elif action_type == 'TYPE' and xpath and value is not None:
                await self.type_text(xpath, str(value))
            elif action_type == 'SELECT' and xpath and value:
                await self.select_option(xpath, value)
            elif action_type == 'PRESS' and value:
                await self.press_key(value)
            elif action_type == 'SCROLL':
                await self.scroll(value or "down")
            elif action_type == 'WAIT':
                try:
                    await asyncio.sleep(int(value) / 1000 if value else 1)
                except (ValueError, TypeError):
                    await asyncio.sleep(1)
            else:
                logger.warning(f"Unknown action type: {action_type}")
            
            # Record action in history
            self._action_history.append(action)
        except Exception as e:
            logger.error(f"Action execution failed: {action_type} - {e}")
            # Record failed action too
            self._action_history.append({**action, 'error': str(e)})
    
    async def _wait_stable(self, timeout: int = 5000):
        """Wait for page to stabilize"""
        try:
            await asyncio.sleep(0.3)
            await self.page.wait_for_load_state('domcontentloaded', timeout=timeout)
        except:
            await asyncio.sleep(0.5)
