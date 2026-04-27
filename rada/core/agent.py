"""
RaDA: Main Agent Orchestrator
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

from ..planner.rad import RaDPlanner
from ..executor.raa import RaAExecutor
from ..verifier.verifier import SubtaskVerifier
from ..retrieval.embedder import Embedder
from ..retrieval.store import ExemplarStore, Exemplar
from ..env.browser_agent import BrowserAgent as BrowserEnvironment
from ..env.dom_parser import DOMParser


class RadaAgent:
    """
    Main RaDA agent that orchestrates task execution
    """
    
    def __init__(
        self,
        llm_model: str = "llama-3.3-70b-versatile",
        embedding_model: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        headless: bool = False,
        exemplar_store_path: str = "./exemplars",
        max_actions_per_subtask: int = 10
    ):
        """
        Initialize the RaDA agent
        
        Args:
            llm_model: LLM model to use
            embedding_model: Embedding model for retrieval
            api_key: API key
            headless: Whether to run browser in headless mode
            exemplar_store_path: Path to store exemplars
            max_actions_per_subtask: Maximum actions per subtask
        """
        self.max_actions_per_subtask = max_actions_per_subtask
        
        # Initialize components
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
        self.embedder = Embedder(model_name=embedding_model)
        self.exemplar_store = ExemplarStore(
            embedder=self.embedder,
            store_path=exemplar_store_path
        )
        self.browser = BrowserEnvironment(headless=headless)
        self.dom_parser = DOMParser()
        
        logger.info("RaDA agent initialized")
    
    async def execute_task(self, task: str, start_url: str) -> Dict[str, Any]:
        """
        Execute a complete task
        
        Args:
            task: Task description
            start_url: Starting URL
            
        Returns:
            Execution result
        """
        logger.info(f"Starting task: {task}")
        
        try:
            # Start browser
            await self.browser.start()
            
            # Navigate to start URL
            page_state = await self.browser.navigate(start_url)
            
            # Step 1: RaD - Decompose task into subtasks
            subtasks = self.rad.decompose(
                task=task,
                url=page_state["url"],
                page_title=page_state["title"],
                page_summary=page_state.get("formatted_elements", "")
            )
            
            logger.info(f"Task decomposed into {len(subtasks)} subtasks")
            
            # Step 2: RaA - Execute each subtask
            all_actions = []
            completed_subtasks = []
            
            for i, subtask in enumerate(subtasks, 1):
                logger.info(f"Executing subtask {i}/{len(subtasks)}: {subtask}")
                
                subtask_result = await self._execute_subtask(
                    task=task,
                    subtask=subtask,
                    remaining_subtasks=subtasks[i:],
                    subtask_index=i
                )
                
                all_actions.extend(subtask_result["actions"])
                completed_subtasks.append(subtask)
                
                # If subtask failed, decide whether to continue
                if not subtask_result["success"]:
                    logger.warning(f"Subtask {i} failed: {subtask_result['reason']}")
                    # Continue with next subtask anyway
            
            # Compile result
            result = {
                "task": task,
                "success": True,
                "subtasks": completed_subtasks,
                "total_actions": len(all_actions),
                "actions": all_actions
            }
            
            logger.info(f"Task completed: {task}")
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "task": task,
                "success": False,
                "error": str(e)
            }
        finally:
            # Stop browser
            await self.browser.stop()
    
    async def _execute_subtask(
        self,
        task: str,
        remaining_subtasks: List[str],
        subtask: str,
        subtask_index: int
    ) -> Dict[str, Any]:
        """
        Execute a single subtask
        
        Args:
            task: Overall task
            subtask: Subtask to execute
            subtask_index: Index of subtask
            
        Returns:
            Subtask execution result
        """
        actions_taken = []
        
        for action_num in range(self.max_actions_per_subtask):
            # Get current page state
            page_state = await self.browser.get_state()
            
            # Extract interactive elements
            elements = self.dom_parser.parse(page_state["html"])
            formatted_elements = self.dom_parser.format_for_llm(elements)
            
            # Retrieve similar exemplars
            exemplars = self.exemplar_store.retrieve(
                query_task=task,
                query_subtask=subtask,
                query_context=page_state["title"]
            )
            formatted_exemplars = self.exemplar_store.format_exemplars(exemplars)
            
            # Generate action using RaA
            action = self.raa.generate_action(
                task=task,
                subtask=subtask,
                remaining_subtasks=remaining_subtasks[1:],
                trajectory=f"Current URL: {page_state['url']}\nPage Title: {page_state['title']}\nElements:\n{formatted_elements[:2000]}",
                action_history=[],
                retrieved_exemplars=formatted_exemplars
            )
            
            logger.info(f"Action {action_num + 1}: {action['action']} - {action.get('reasoning', '')}")
            
            # Execute action
            try:
                await self.browser.execute_action(action)
                actions_taken.append({
                    "action": action,
                    "result": "success"
                })
            except Exception as e:
                logger.error(f"Action execution failed: {e}")
                actions_taken.append({
                    "action": action,
                    "result": f"failed: {str(e)}"
                })
            
            
            # Verify if subtask is complete
            new_state = await self.browser.get_state()
            verification = self.verifier.verify(
                subtask=subtask,
                page_title=new_state["title"],
                url=new_state["url"],
                page_summary=formatted_elements[:500],
                previous_action=action,
                subtasks=remaining_subtasks,
                completed_subtasks=[],
                action_history=actions_taken,
                trajectory=f"Current URL: {new_state['url']}\nPage Title: {new_state['title']}"
            )
            
            if verification["completed"] and verification["confidence"] > 0.7:
                logger.info(f"Subtask completed successfully")
                
                # Save successful trajectory as exemplar
                exemplar = Exemplar(
                    task=task,
                    subtask=subtask,
                    context=page_state["title"],
                    actions=[a["action"] for a in actions_taken],
                    outcome="success"
                )
                self.exemplar_store.add_exemplar(exemplar)
                
                return {
                    "success": True,
                    "actions": actions_taken,
                    "reason": "Subtask completed"
                }
        
        # Max actions reached without completion
        logger.warning(f"Max actions reached for subtask: {subtask}")
        return {
            "success": False,
            "actions": actions_taken,
            "reason": "Max actions reached"
        }
    

    
    async def add_exemplar_manually(
        self,
        task: str,
        subtask: str,
        actions: List[Dict],
        outcome: str,
        context: str = ""
    ):
        """
        Manually add an exemplar to the store
        
        Args:
            task: Task description
            subtask: Subtask description
            actions: List of actions
            outcome: Outcome description
            context: Additional context
        """
        exemplar = Exemplar(
            task=task,
            subtask=subtask,
            context=context,
            actions=actions,
            outcome=outcome
        )
        self.exemplar_store.add_exemplar(exemplar)
        logger.info("Manually added exemplar")
