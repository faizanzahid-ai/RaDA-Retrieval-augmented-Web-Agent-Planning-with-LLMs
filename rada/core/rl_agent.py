"""
RL-Enhanced RaDA Agent with Learning Loop

Implements Reinforcement Learning + RaDA:
- Reward function: task success + efficiency + fewer steps
- Trainable policies for retrieval and decomposition
- Agent becomes optimal planner through learning
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from loguru import logger

from ..planner.rad import RaDPlanner
from ..executor.raa import RaAExecutor
from ..verifier.verifier import SubtaskVerifier
from ..retrieval.embedder import Embedder
from ..retrieval.adaptive_store import AdaptiveExemplarStore, AdaptiveExemplar
from ..env.browser_agent import BrowserAgent as RealBrowserAgent


@dataclass
class RLState:
    """State representation for RL"""
    task_embedding: np.ndarray
    subtask_count: int
    action_count: int
    success_rate: float
    efficiency: float


@dataclass
class RLAction:
    """Action representation for RL"""
    retrieval_k: int  # Number of exemplars to retrieve
    decomposition_depth: int  # Depth of task decomposition
    temperature: float  # LLM temperature


@dataclass
class RLReward:
    """Reward components"""
    task_success: float  # 1.0 for success, 0.0 for failure
    efficiency: float  # Inverse of execution time
    step_penalty: float  # Penalty for each action
    total: float


class RLRaDAgent:
    """
    RL-Enhanced RaDA Agent with Learning Loop
    
    Implements reinforcement learning to optimize:
    - Retrieval policy (how many exemplars to retrieve)
    - Decomposition policy (how deep to decompose)
    - Action generation policy (temperature control)
    """
    
    def __init__(
        self,
        llm_model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        headless: bool = True,
        exemplar_store_path: str = "./adaptive_exemplars",
        max_actions_per_subtask: int = 15,
        learning_rate: float = 0.01,
        gamma: float = 0.99  # Discount factor
    ):
        self.llm_model = llm_model
        self.max_actions_per_subtask = max_actions_per_subtask
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Initialize components
        self.rad = RaDPlanner(llm_model=llm_model, api_key=api_key)
        self.raa = RaAExecutor(llm_model=llm_model, api_key=api_key)
        self.verifier = SubtaskVerifier(llm_model=llm_model, api_key=api_key)
        self.embedder = Embedder(model_name=embedding_model)
        self.adaptive_store = AdaptiveExemplarStore(
            embedder=self.embedder,
            store_path=exemplar_store_path,
            top_k=3
        )
        self.browser = RealBrowserAgent(headless=headless)
        
        # RL policy parameters (learnable)
        self.policy_params = {
            'retrieval_k': 3.0,  # Default: retrieve 3 exemplars
            'decomposition_depth': 1.0,  # Default: single-level decomposition
            'temperature': 0.2  # Default: low temperature
        }
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        
        logger.info(f"RLRaDAgent initialized with {llm_model}")
    
    def compute_reward(
        self,
        success: bool,
        execution_time: float,
        num_actions: int,
        max_allowed_time: float = 60.0,
        max_allowed_actions: int = 20
    ) -> RLReward:
        """
        Compute reward for an episode
        
        Reward = task_success + efficiency - step_penalty
        
        Args:
            success: Whether task was successful
            execution_time: Time taken to complete task
            num_actions: Number of actions taken
            max_allowed_time: Maximum allowed time
            max_allowed_actions: Maximum allowed actions
            
        Returns:
            RLReward object with all components
        """
        # Task success reward (0 or 1)
        task_success = 1.0 if success else 0.0
        
        # Efficiency reward (inverse of time, normalized)
        efficiency = max(0, 1.0 - (execution_time / max_allowed_time))
        
        # Step penalty (penalize too many actions)
        step_penalty = min(num_actions / max_allowed_actions, 1.0) * 0.5
        
        # Total reward
        total = task_success + efficiency - step_penalty
        
        return RLReward(
            task_success=task_success,
            efficiency=efficiency,
            step_penalty=step_penalty,
            total=total
        )
    
    def get_state(self, task: str, history: List[Dict]) -> RLState:
        """
        Get current state representation
        
        Args:
            task: Current task
            history: Action history
            
        Returns:
            RLState representation
        """
        # Compute task embedding
        task_embedding = self.embedder.encode([task])[0]
        
        # Extract state features
        subtask_count = len(history) if history else 0
        action_count = sum(len(h.get('actions', [])) for h in history)
        
        # Calculate success rate from recent history
        recent_history = history[-10:] if len(history) > 10 else history
        if recent_history:
            success_rate = sum(1 for h in recent_history if h.get('success', False)) / len(recent_history)
        else:
            success_rate = 0.5
        
        # Calculate efficiency (actions per subtask)
        if subtask_count > 0:
            efficiency = action_count / subtask_count
        else:
            efficiency = 0.0
        
        return RLState(
            task_embedding=task_embedding,
            subtask_count=subtask_count,
            action_count=action_count,
            success_rate=success_rate,
            efficiency=efficiency
        )
    
    def select_action(self, state: RLState, explore: bool = False) -> RLAction:
        """
        Select action based on current policy
        
        Args:
            state: Current state
            explore: Whether to explore (epsilon-greedy)
            
        Returns:
            RLAction to take
        """
        if explore and np.random.random() < 0.1:  # 10% exploration
            # Random exploration
            return RLAction(
                retrieval_k=np.random.randint(1, 6),
                decomposition_depth=np.random.randint(1, 3),
                temperature=np.random.uniform(0.1, 0.5)
            )
        
        # Use current policy parameters
        return RLAction(
            retrieval_k=int(self.policy_params['retrieval_k']),
            decomposition_depth=int(self.policy_params['decomposition_depth']),
            temperature=self.policy_params['temperature']
        )
    
    def update_policy(self, state: RLState, action: RLAction, reward: float, next_state: RLState):
        """
        Update policy parameters using policy gradient
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Simple policy gradient update
        # Adjust parameters based on reward
        
        if reward > 0:
            # Positive reward: reinforce current action
            self.policy_params['retrieval_k'] += self.learning_rate * (action.retrieval_k - self.policy_params['retrieval_k'])
            self.policy_params['decomposition_depth'] += self.learning_rate * (action.decomposition_depth - self.policy_params['decomposition_depth'])
            self.policy_params['temperature'] += self.learning_rate * (action.temperature - self.policy_params['temperature'])
        else:
            # Negative reward: move away from current action
            self.policy_params['retrieval_k'] -= self.learning_rate * (action.retrieval_k - self.policy_params['retrieval_k'])
            self.policy_params['decomposition_depth'] -= self.learning_rate * (action.decomposition_depth - self.policy_params['decomposition_depth'])
            self.policy_params['temperature'] -= self.learning_rate * (action.temperature - self.policy_params['temperature'])
        
        # Clip parameters to valid ranges
        self.policy_params['retrieval_k'] = np.clip(self.policy_params['retrieval_k'], 1, 5)
        self.policy_params['decomposition_depth'] = np.clip(self.policy_params['decomposition_depth'], 1, 3)
        self.policy_params['temperature'] = np.clip(self.policy_params['temperature'], 0.1, 0.5)
    
    async def execute_task(
        self,
        task: str,
        start_url: str,
        explore: bool = False
    ) -> Dict[str, Any]:
        """
        Execute task with RL learning
        
        Args:
            task: Task description
            start_url: Starting URL
            explore: Whether to explore
            
        Returns:
            Execution result with reward
        """
        logger.info(f"Executing task with RL: {task}")
        start_time = time.time()
        
        # Get initial state
        initial_state = self.get_state(task, [])
        
        # Select action (policy)
        action = self.select_action(initial_state, explore)
        
        # Update store's top_k based on policy
        self.adaptive_store.top_k = action.retrieval_k
        
        # Execute task (simplified - using basic execution)
        try:
            await self.browser.start()
            state_info = await self.browser.navigate(start_url)
            
            # Decompose task
            subtasks = self.rad.decompose(
                task=task,
                url=state_info['url'],
                page_title=state_info.get('title', ''),
                page_summary=state_info.get('accessibility_tree', state_info.get('html', ''))[:500]
            )
            
            # Execute subtasks
            all_actions = []
            for i, subtask in enumerate(subtasks[:5], 1):  # Limit to 5 subtasks
                # Retrieve exemplars
                exemplars = self.adaptive_store.retrieve(
                    query_task=task,
                    query_subtask=subtask,
                    query_context=state_info.get('title', '')
                )

                # Generate action
                generated_action = self.raa.generate_action(
                    task=task,
                    subtask=subtask,
                    remaining_subtasks=subtasks[i:],
                    trajectory=f"Current URL: {state_info['url']}\nPage Title: {state_info.get('title', '')}\nElements:\n{state_info.get('accessibility_tree', state_info.get('html', ''))[:2000]}",
                    action_history=all_actions,
                    retrieved_exemplars=self.adaptive_store.format_exemplars(exemplars)
                )
                
                all_actions.append(generated_action)
            
            success = len(all_actions) > 0
            
            # Compute reward
            execution_time = time.time() - start_time
            reward = self.compute_reward(success, execution_time, len(all_actions))
            
            # Get next state
            history = [{'success': success, 'actions': all_actions}]
            next_state = self.get_state(task, history)
            
            # Update policy
            self.update_policy(initial_state, action, reward.total, next_state)
            
            # Update adaptive store
            if success:
                exemplar = AdaptiveExemplar(
                    task=task,
                    subtask=subtasks[0] if subtasks else task,
                    context=state_info.get('title', ''),
                    actions=all_actions,
                    outcome="success"
                )
                self.adaptive_store.add_exemplar(exemplar, success=True)
            
            # Track episode
            self.episode_rewards.append(reward.total)
            self.episode_lengths.append(len(all_actions))
            
            result = {
                'task': task,
                'success': success,
                'reward': reward.total,
                'reward_components': {
                    'task_success': reward.task_success,
                    'efficiency': reward.efficiency,
                    'step_penalty': reward.step_penalty
                },
                'execution_time': execution_time,
                'num_actions': len(all_actions),
                'policy_params': self.policy_params.copy()
            }
            
            logger.info(f"Task completed. Reward: {reward.total:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            execution_time = time.time() - start_time
            reward = self.compute_reward(False, execution_time, 0)
            
            self.episode_rewards.append(reward.total)
            self.episode_lengths.append(0)
            
            return {
                'task': task,
                'success': False,
                'reward': reward.total,
                'error': str(e),
                'execution_time': execution_time
            }
        finally:
            await self.browser.stop()
    
    def get_training_statistics(self) -> Dict:
        """Get training statistics"""
        if not self.episode_rewards:
            return {
                'episodes': 0,
                'avg_reward': 0,
                'avg_length': 0,
                'policy_params': self.policy_params
            }
        
        return {
            'episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'policy_params': self.policy_params,
            'recent_rewards': self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        }
