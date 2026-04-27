"""
Real RaDA Agent with Browser Automation

Implements the complete RaDA pipeline with:
- Real browser control via Playwright
- DOM parsing and XPath grounding
- Retrieval-augmented task decomposition (RaD)
- Retrieval-augmented action generation (RaA)
- Full action execution loop
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from ..planner.rad import RaDPlanner
from ..executor.raa import RaAExecutor
from ..verifier.verifier import SubtaskVerifier
from ..retrieval.embedder import Embedder
from ..retrieval.store import ExemplarStore, Exemplar
from ..env.browser_agent import BrowserAgent as RealBrowserAgent


class RealRaDAAgent:
    """
    Real RaDA agent with browser automation
    
    Implements the complete two-stage planning process:
    1. RaD: Task decomposition into subtasks
    2. RaA: Action generation with exemplar retrieval
    """
    
    def __init__(
        self,
        llm_model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        headless: bool = True,
        exemplar_store_path: str = "./exemplars",
        max_actions_per_subtask: int = 15,
        max_subtasks: int = 7
    ):
        """
        Initialize real RaDA agent
        
        Args:
            llm_model: LLM model for planning
            api_key: API key for LLM
            embedding_model: Embedding model for retrieval
            headless: Browser headless mode
            exemplar_store_path: Path to store exemplars
            max_actions_per_subtask: Maximum actions per subtask
            max_subtasks: Maximum number of subtasks
        """
        self.max_actions_per_subtask = max_actions_per_subtask
        self.max_subtasks = max_subtasks
        
        # Initialize RaD and RaA
        self.rad = RaDPlanner(
            llm_model=llm_model,
            api_key=api_key
        )
        self.raa = RaAExecutor(
            llm_model=llm_model,
            api_key=api_key
        )
        self.verifier = SubtaskVerifier(
            llm_model=llm_model,
            api_key=api_key
        )
        
        # Initialize retrieval system
        self.embedder = Embedder(model_name=embedding_model)
        self.exemplar_store = ExemplarStore(
            embedder=self.embedder,
            store_path=exemplar_store_path,
            top_k=3
        )
        
        # Initialize browser agent
        self.browser = RealBrowserAgent(headless=headless)
        
        logger.info(f"RealRaDAAgent initialized with {llm_model}")
    
    async def execute_task(
        self,
        task: str,
        start_url: str,
        save_screenshots: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a complete web task
        
        Args:
            task: Task description
            start_url: Starting URL
            save_screenshots: Whether to save screenshots
            
        Returns:
            Task execution result
        """
        logger.info(f"Starting task: {task}")
        start_time = time.time()
        
        try:
            # Start browser
            await self.browser.start()
            
            # Navigate to start URL with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    state = await self.browser.navigate(start_url)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Navigation attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(2)
            
            if save_screenshots:
                await self.browser.take_screenshot(f"screenshots/start_{int(time.time())}.png")
            
            # Stage 1: RaD - Task Decomposition
            logger.info("Stage 1: Task Decomposition (RaD)")
            subtasks = self.rad.decompose(
                task=task,
                url=state['url'],
                page_title=state['title'],
                page_summary=state.get('accessibility_tree', state.get('html', ''))[:500]
            )
            
            # Limit subtasks
            subtasks = subtasks[:self.max_subtasks]
            logger.info(f"Decomposed into {len(subtasks)} subtasks")
            
            # Stage 2: RaA - Action Generation & Execution
            logger.info("Stage 2: Action Generation (RaA)")
            all_actions = []
            completed_subtasks = []
            failed_subtasks = []
            
            for i, subtask in enumerate(subtasks, 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Subtask {i}/{len(subtasks)}: {subtask}")
                logger.info(f"{'='*60}")
                
                subtask_result = await self._execute_subtask(
                    task=task,
                    subtask=subtask,
                    remaining_subtasks=subtasks[i:],
                    subtask_index=i,
                    save_screenshot=save_screenshots
                )
                
                if subtask_result['success']:
                    completed_subtasks.append(subtask)
                    all_actions.extend(subtask_result['actions'])
                    
                    # Save successful exemplar
                    self._save_exemplar(task, subtask, subtask_result['actions'])
                else:
                    failed_subtasks.append({
                        'subtask': subtask,
                        'reason': subtask_result.get('reason', 'Unknown')
                    })
                    logger.warning(f"Subtask failed: {subtask_result.get('reason')}")
            
            execution_time = time.time() - start_time
            
            # Compile result
            result = {
                'task': task,
                'start_url': start_url,
                'success': len(failed_subtasks) == 0,
                'subtasks_total': len(subtasks),
                'subtasks_completed': len(completed_subtasks),
                'subtasks_failed': len(failed_subtasks),
                'completed_subtasks': completed_subtasks,
                'failed_subtasks': failed_subtasks,
                'total_actions': len(all_actions),
                'actions': all_actions,
                'execution_time': execution_time,
                'action_history': self.browser.get_action_history()
            }
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Task completed: {task}")
            logger.info(f"Success: {result['success']}")
            logger.info(f"Time: {execution_time:.2f}s")
            logger.info(f"Actions: {len(all_actions)}")
            logger.info(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task execution failed: {e}")
            
            return {
                'task': task,
                'start_url': start_url,
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'actions': []
            }
        finally:
            await self.browser.stop()
    
    async def _execute_subtask(
        self,
        task: str,
        subtask: str,
        remaining_subtasks: List[str],
        subtask_index: int,
        save_screenshot: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a single subtask
        
        Args:
            task: Overall task
            subtask: Subtask to execute
            subtask_index: Subtask index
            save_screenshot: Save screenshots
            
        Returns:
            Subtask execution result
        """
        actions_taken = []
        
        for action_num in range(self.max_actions_per_subtask):
            try:
                # Observe current state
                state = await self.browser.observe()  # Returns dict with url, title, html, accessibility_tree
                
                # Retrieve exemplars
                exemplars = self.exemplar_store.retrieve(
                    query_task=task,
                    query_subtask=subtask,
                    query_context=state['title']
                )
                formatted_exemplars = self.exemplar_store.format_exemplars(exemplars)
                
                # Generate action using RaA
                action = self.raa.generate_action(
                    task=task,
                    subtask=subtask,
                    remaining_subtasks=remaining_subtasks[1:],
                    trajectory=f"Current URL: {state['url']}\nPage Title: {state['title']}\nElements:\n{state.get('accessibility_tree', state.get('html', ''))[:2000]}",
                    action_history=actions_taken,
                    retrieved_exemplars=formatted_exemplars
                )
                
                logger.info(f"Action {action_num + 1}: {action['action']} - {action.get('reasoning', '')}")
                
                # Execute action
                await self.browser.execute_action(action)
                new_state = await self.browser.observe()
                
                actions_taken.append({
                    'action': action,
                    'state_before': {
                        'url': state['url'],
                        'title': state['title']
                    },
                    'state_after': {
                        'url': new_state['url'],
                        'title': new_state['title']
                    }
                })
                
                if save_screenshot:
                    await self.browser.take_screenshot(
                        f"screenshots/subtask{subtask_index}_action{action_num + 1}.png"
                    )
                
                # Verify subtask completion
                verification = self.verifier.verify(
                    subtask=subtask,
                    page_title=new_state['title'],
                    url=new_state['url'],
                    page_summary=new_state.get('accessibility_tree', new_state.get('html', ''))[:500],
                    previous_action=action,
                    subtasks=remaining_subtasks,
                    completed_subtasks=[],
                    action_history=actions_taken,
                    trajectory=f"Current URL: {new_state['url']}\nPage Title: {new_state['title']}"
                )
                
                if verification['completed'] and verification['confidence'] > 0.7:
                    logger.info(f"✓ Subtask completed successfully")
                    
                    return {
                        'success': True,
                        'actions': actions_taken,
                        'num_actions': len(actions_taken),
                        'reason': 'Subtask completed'
                    }
                
            except Exception as e:
                logger.error(f"Action execution failed: {e}")
                actions_taken.append({
                    'action': None,
                    'error': str(e)
                })
        
        # Max actions reached
        logger.warning(f"✗ Max actions ({self.max_actions_per_subtask}) reached")
        return {
            'success': False,
            'actions': actions_taken,
            'num_actions': len(actions_taken),
            'reason': 'Max actions reached'
        }
    
    def _save_exemplar(self, task: str, subtask: str, actions: List[Dict]):
        """
        Save successful trajectory as exemplar
        
        Args:
            task: Task description
            subtask: Subtask description
            actions: Actions taken
        """
        exemplar = Exemplar(
            task=task,
            subtask=subtask,
            context=f"Successfully completed {subtask}",
            actions=[a['action'] for a in actions if a.get('action')],
            outcome="success"
        )
        
        self.exemplar_store.add_exemplar(exemplar)
        logger.info(f"Saved exemplar for: {subtask}")
