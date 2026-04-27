"""
Main RaDA Loop

Complete RaDA agent with:
- RaD: Task decomposition
- RaA: Action generation  
- Retrieval augmentation
- Verification
- Browser execution
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from loguru import logger
from rich.console import Console

from rada.retriever.retriever import TrajectoryRetriever
from rada.planner.rad import RaDPlanner
from rada.executor.raa import RaAExecutor
from rada.verifier.verifier import SubtaskVerifier
from rada.env.web_env import WebEnvironment
from rada.config import Config
from rada.utils.openrouter_client import OpenRouterClient
from rada.utils.nvidia_client import NVIDIAClient
from rada.utils.groq_client import GroqClient

console = Console()


class RADA:
    """
    Complete RaDA Agent
    
    Implements the full retrieval-augmented web agent planning loop
    """
    
    def __init__(
        self,
        llm_model: str = None,
        llm_provider: str = None,
        headless: bool = True,
        max_actions_per_subtask: int = 8
    ):
        self.max_actions = max_actions_per_subtask
        
        # Get model and provider from config if not specified
        if llm_model is None:
            llm_model = Config.get_available_model()
        if llm_provider is None:
            llm_provider = Config.LLM_PROVIDER
        
        self.llm_provider = llm_provider
        print(f"Using model: {llm_model} ({llm_provider.upper()})")
        
        # Initialize LLM client based on provider
        if llm_provider.lower() == "openrouter":
            self.llm_client = OpenRouterClient(model=llm_model)
        elif llm_provider.lower() == "nvidia":
            self.llm_client = NVIDIAClient(model=llm_model)
        else:  # groq
            self.llm_client = GroqClient(model=llm_model)
        
        # Initialize components with benchmark parameter
        self.retriever = TrajectoryRetriever()
        self.planner = RaDPlanner(llm_client=self.llm_client, benchmark="compwob")
        self.executor = RaAExecutor(llm_client=self.llm_client, benchmark="compwob")
        self.verifier = SubtaskVerifier(llm_client=self.llm_client, benchmark="compwob")
        self.env = WebEnvironment(headless=headless)
    
    async def execute_task(self, task: str, start_url: str) -> Dict[str, Any]:
        """
        Execute complete task
        
        Args:
            task: Task description
            start_url: Starting URL
            
        Returns:
            Execution result
        """
        console.print(f"\n[bold cyan]Task: {task}[/bold cyan]\n")
        start_time = time.time()
        
        try:
            # Start environment
            await self.env.start()
            
            # Navigate
            state = await self.env.navigate(start_url)
            
            # RaD: Decompose task
            console.print("[yellow]RaD: Decomposing task...[/yellow]")
            similar_tasks = self.retriever.retrieve(task, "", state['title'])
            subtasks = self.planner.decompose(
                task=task,
                url=state['url'],
                page_title=state['title'],
                page_summary=state.get('formatted_elements', ''),
                similar_tasks=similar_tasks
            )
            
            console.print(f"[green]✓ Decomposed into {len(subtasks)} subtasks[/green]\n")
            
            # RaA: Execute subtasks
            all_actions = []
            completed_subtasks = []
            
            for i, subtask in enumerate(subtasks, 1):
                console.print(f"[blue]Subtask {i}/{len(subtasks)}: {subtask}[/blue]")
                
                subtask_result = await self._execute_subtask(
                    task=task,
                    subtask=subtask,
                    subtask_idx=i,
                    subtasks=subtasks
                )
                
                if subtask_result['success']:
                    completed_subtasks.append(subtask)
                    all_actions.extend(subtask_result['actions'])
                    
                    # Save successful trajectory
                    self._save_trajectory(task, subtask, subtask_result['actions'])
                
                console.print()
            
            execution_time = time.time() - start_time
            
            return {
                'task': task,
                'success': len(completed_subtasks) == len(subtasks),
                'subtasks_completed': len(completed_subtasks),
                'subtasks_total': len(subtasks),
                'total_actions': len(all_actions),
                'actions': all_actions,
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"Task failed: {e}")
            return {
                'task': task,
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
        finally:
            try:
                await self.env.stop()
            except Exception as e:
                logger.warning(f"Browser cleanup warning: {e}")
                pass  # Ignore cleanup errors
    
    async def _execute_subtask(
        self,
        task: str,
        subtask: str,
        subtask_idx: int,
        subtasks: List[str]
    ) -> Dict[str, Any]:
        """Execute single subtask"""
        actions_taken = []
        recent_actions = []  # Track recent actions to detect loops
        
        for action_num in range(self.max_actions):
            # Observe
            state = await self.env._observe()
            
            # Retrieve exemplars
            exemplars = self.retriever.retrieve(task, subtask, state['title'])
            formatted_exemplars = self.retriever.format_for_prompt(exemplars)
            
            # Generate action using new paper-specific signature
            remaining_subtasks = [subtask]  # Simplified for current subtask
            trajectory = f"Current URL: {state['url']}\nPage Title: {state['title']}\nElements:\n{state['formatted_elements'][:2000]}"
            
            action = self.executor.generate_action(
                task=task,
                subtask=subtask,
                remaining_subtasks=remaining_subtasks,
                trajectory=trajectory,
                action_history=actions_taken,
                retrieved_exemplars=formatted_exemplars
            )
            
            # Check for duplicate actions (loop detection)
            action_key = f"{action['action']}:{action.get('element_xpath', '')}:{action.get('value', '')}"
            if action_key in recent_actions:
                console.print(f"  [yellow]⚠ Duplicate action detected, skipping[/yellow]")
                continue
            
            recent_actions.append(action_key)
            if len(recent_actions) > 3:  # Keep last 3 actions
                recent_actions.pop(0)
            
            console.print(f"  Action {action_num + 1}: {action['action']} - {action['reasoning']}")
            
            try:
                # Execute
                new_state = await self.env.execute_action(action)
                actions_taken.append(action)
                
                # Verify with new signature
                verification = self.verifier.verify(
                    subtask=subtask,
                    page_title=new_state['title'],
                    url=new_state['url'],
                    page_summary=new_state.get('formatted_elements', ''),
                    previous_action=action,
                    subtasks=subtasks,
                    completed_subtasks=[],
                    action_history=actions_taken,
                    trajectory=f"Current URL: {new_state['url']}\nPage Title: {new_state['title']}"
                )
                
                if verification['completed'] and verification['confidence'] > 0.7:
                    console.print(f"  [green]✓ Subtask completed[/green]")
                    return {
                        'success': True,
                        'actions': actions_taken,
                        'reason': 'Completed'
                    }
            except Exception as e:
                logger.error(f"Action execution failed: {e}")
                console.print(f"  [red]✗ Action failed: {str(e)[:100]}[/red]")
                continue
        
        console.print(f"  [red]✗ Max actions reached[/red]")
        return {
            'success': False,
            'actions': actions_taken,
            'reason': 'Max actions'
        }
    
    def _save_trajectory(self, task: str, subtask: str, actions: List[Dict]):
        """Save successful trajectory"""
        trajectory = {
            'id': f"traj_{len(self.retriever.trajectories) + 1:04d}",
            'task': task,
            'subtask': subtask,
            'actions': actions,
            'outcome': 'success'
        }
        
        self.retriever.add_trajectory(trajectory)
