"""
Extended RaDA Evaluation Metrics

Comprehensive metrics for evaluating the three novelties:
1. Adaptive Memory System
2. Graph-Based Task Decomposition
3. RL + RaDA Learning Loop
"""

from typing import List, Dict, Any
import numpy as np
from loguru import logger


class ExtendedRaDAMetrics:
    """Metrics for extended RaDA evaluation"""
    
    def __init__(self):
        self.adaptive_memory_metrics = {}
        self.graph_metrics = {}
        self.rl_metrics = {}
    
    def compute_adaptive_memory_metrics(self, store_statistics: Dict) -> Dict:
        """Compute metrics for adaptive memory system"""
        return {
            'total_exemplars': store_statistics.get('total_exemplars', 0),
            'avg_success_rate': store_statistics.get('avg_success_rate', 0),
            'avg_adaptive_score': store_statistics.get('avg_adaptive_score', 0),
            'high_performers_ratio': store_statistics.get('high_performers', 0) / max(store_statistics.get('total_exemplars', 1), 1)
        }
    
    def compute_graph_metrics(self, graph: Dict) -> Dict:
        """Compute metrics for graph-based decomposition"""
        subtasks = graph.get('subtasks', [])
        edges = graph.get('edges', [])
        
        # Calculate parallelization potential
        parallel_groups = self._get_parallel_groups(graph)
        max_parallel = max(len(g) for g in parallel_groups) if parallel_groups else 1
        
        return {
            'num_subtasks': len(subtasks),
            'num_dependencies': len(edges),
            'max_parallel_subtasks': max_parallel,
            'parallelization_ratio': max_parallel / max(len(subtasks), 1),
            'avg_dependencies_per_subtask': len(edges) / max(len(subtasks), 1)
        }
    
    def _get_parallel_groups(self, graph: Dict) -> List[List]:
        """Get parallel execution groups from graph"""
        subtasks = graph.get('subtasks', [])
        groups = []
        completed = set()
        
        while len(completed) < len(subtasks):
            ready = []
            for st in subtasks:
                if st['id'] not in completed:
                    deps = st.get('dependencies', [])
                    if all(d in completed for d in deps):
                        ready.append(st)
            
            if not ready:
                break
            
            groups.append(ready)
            completed.update(st['id'] for st in ready)
        
        return groups
    
    def compute_rl_metrics(self, episode_rewards: List[float], episode_lengths: List[int]) -> Dict:
        """Compute RL learning metrics"""
        if not episode_rewards:
            return {}
        
        return {
            'total_episodes': len(episode_rewards),
            'avg_reward': np.mean(episode_rewards),
            'reward_std': np.std(episode_rewards),
            'reward_improvement': episode_rewards[-1] - episode_rewards[0] if len(episode_rewards) > 1 else 0,
            'avg_episode_length': np.mean(episode_lengths),
            'learning_stability': 1.0 - (np.std(episode_rewards) / (np.mean(episode_rewards) + 1e-8))
        }
    
    def compute_comprehensive_metrics(
        self,
        original_results: List[Dict],
        extended_results: List[Dict],
        store_stats: Dict = None,
        graph_data: Dict = None
    ) -> Dict:
        """Compute comprehensive comparison metrics"""
        # Original metrics
        orig_success_rate = np.mean([r.get('success', False) for r in original_results])
        orig_avg_time = np.mean([r.get('execution_time', 0) for r in original_results])
        orig_avg_actions = np.mean([r.get('num_actions', 0) for r in original_results])
        
        # Extended metrics
        ext_success_rate = np.mean([r.get('success', False) for r in extended_results])
        ext_avg_time = np.mean([r.get('execution_time', 0) for r in extended_results])
        ext_avg_actions = np.mean([r.get('num_actions', 0) for r in extended_results])
        ext_avg_reward = np.mean([r.get('reward', 0) for r in extended_results])
        
        # Improvements
        success_improvement = (ext_success_rate - orig_success_rate) / max(orig_success_rate, 0.01) * 100
        time_improvement = (orig_avg_time - ext_avg_time) / max(orig_avg_time, 0.01) * 100
        action_improvement = (orig_avg_actions - ext_avg_actions) / max(orig_avg_actions, 0.01) * 100
        
        metrics = {
            'original': {
                'success_rate': orig_success_rate,
                'avg_execution_time': orig_avg_time,
                'avg_actions': orig_avg_actions
            },
            'extended': {
                'success_rate': ext_success_rate,
                'avg_execution_time': ext_avg_time,
                'avg_actions': ext_avg_actions,
                'avg_reward': ext_avg_reward
            },
            'improvements': {
                'success_rate_pct': success_improvement,
                'time_reduction_pct': time_improvement,
                'action_reduction_pct': action_improvement
            }
        }
        
        # Add novelty-specific metrics
        if store_stats:
            metrics['adaptive_memory'] = self.compute_adaptive_memory_metrics(store_stats)
        
        if graph_data:
            metrics['graph_decomposition'] = self.compute_graph_metrics(graph_data)
        
        return metrics
    
    def print_metrics_report(self, metrics: Dict):
        """Print comprehensive metrics report"""
        logger.info("="*60)
        logger.info("EXTENDED RaDA EVALUATION METRICS")
        logger.info("="*60)
        
        logger.info("\n[ORIGINAL RaDA]")
        logger.info(f"  Success Rate: {metrics['original']['success_rate']:.2%}")
        logger.info(f"  Avg Execution Time: {metrics['original']['avg_execution_time']:.2f}s")
        logger.info(f"  Avg Actions: {metrics['original']['avg_actions']:.1f}")
        
        logger.info("\n[EXTENDED RaDA]")
        logger.info(f"  Success Rate: {metrics['extended']['success_rate']:.2%}")
        logger.info(f"  Avg Execution Time: {metrics['extended']['avg_execution_time']:.2f}s")
        logger.info(f"  Avg Actions: {metrics['extended']['avg_actions']:.1f}")
        logger.info(f"  Avg Reward: {metrics['extended']['avg_reward']:.3f}")
        
        logger.info("\n[IMPROVEMENTS]")
        logger.info(f"  Success Rate: {metrics['improvements']['success_rate_pct']:+.1f}%")
        logger.info(f"  Time Reduction: {metrics['improvements']['time_reduction_pct']:+.1f}%")
        logger.info(f"  Action Reduction: {metrics['improvements']['action_reduction_pct']:+.1f}%")
        
        if 'adaptive_memory' in metrics:
            logger.info("\n[ADAPTIVE MEMORY]")
            am = metrics['adaptive_memory']
            logger.info(f"  Total Exemplars: {am['total_exemplars']}")
            logger.info(f"  Avg Success Rate: {am['avg_success_rate']:.2%}")
            logger.info(f"  Avg Adaptive Score: {am['avg_adaptive_score']:.3f}")
            logger.info(f"  High Performers Ratio: {am['high_performers_ratio']:.2%}")
        
        if 'graph_decomposition' in metrics:
            logger.info("\n[GRAPH DECOMPOSITION]")
            gd = metrics['graph_decomposition']
            logger.info(f"  Subtasks: {gd['num_subtasks']}")
            logger.info(f"  Dependencies: {gd['num_dependencies']}")
            logger.info(f"  Max Parallel: {gd['max_parallel_subtasks']}")
            logger.info(f"  Parallelization Ratio: {gd['parallelization_ratio']:.2%}")
        
        logger.info("="*60)
