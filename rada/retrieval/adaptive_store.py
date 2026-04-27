"""
Adaptive Exemplar Store with Lifelong Learning

Implements "Lifelong Retrieval-Augmented Planning":
- Memory evolves after each task
- Compress → store → rank → re-weight
- Penalize bad exemplars after failure
- Memory Score = Success Rate × Recency × Diversity
"""

import os
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from loguru import logger

from .embedder import Embedder


@dataclass
class AdaptiveExemplar:
    """Exemplar with adaptive scoring"""
    task: str
    subtask: str
    context: str
    actions: List[Dict]
    outcome: str
    embedding: Optional[List[float]] = None
    success_count: int = 0
    failure_count: int = 0
    last_used: float = 0.0
    created_at: float = 0.0
    diversity_score: float = 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AdaptiveExemplar':
        return cls(**data)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Default for new exemplars
        return self.success_count / total
    
    @property
    def recency_score(self) -> float:
        """Calculate recency score (exponential decay)"""
        if self.last_used == 0:
            return 0.5
        age = time.time() - self.last_used
        # Decay over 30 days (2592000 seconds)
        decay = np.exp(-age / 2592000)
        return max(0.1, decay)
    
    @property
    def adaptive_score(self) -> float:
        """Memory Score = Success Rate × Recency × Diversity"""
        return self.success_rate * self.recency_score * self.diversity_score


class AdaptiveExemplarStore:
    """Adaptive exemplar store with lifelong learning"""
    
    def __init__(
        self,
        embedder: Embedder,
        store_path: str = "./adaptive_exemplars",
        top_k: int = 3,
        diversity_threshold: float = 0.7
    ):
        """
        Initialize adaptive exemplar store
        
        Args:
            embedder: Embedder instance
            store_path: Path to store exemplars
            top_k: Number of exemplars to retrieve
            diversity_threshold: Threshold for diversity scoring
        """
        self.embedder = embedder
        self.store_path = Path(store_path)
        self.top_k = top_k
        self.diversity_threshold = diversity_threshold
        self.exemplars: List[AdaptiveExemplar] = []
        self.embeddings: Optional[np.ndarray] = None
        
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._load_exemplars()
    
    def _load_exemplars(self):
        """Load exemplars from disk"""
        exemplar_file = self.store_path / "adaptive_exemplars.json"
        if exemplar_file.exists():
            with open(exemplar_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.exemplars = [AdaptiveExemplar.from_dict(e) for e in data]
            logger.info(f"Loaded {len(self.exemplars)} adaptive exemplars from disk")
            
            # Load or compute embeddings
            embeddings_file = self.store_path / "adaptive_embeddings.npy"
            if embeddings_file.exists():
                self.embeddings = np.load(embeddings_file)
                logger.info(f"Loaded embeddings for {len(self.exemplars)} exemplars")
            else:
                self._compute_embeddings()
        else:
            logger.info("No adaptive exemplars found on disk, starting with empty store")
    
    def _compute_embeddings(self):
        """Compute embeddings for all exemplars"""
        if not self.exemplars:
            self.embeddings = np.array([])
            return
        
        texts = [f"{e.task} {e.subtask} {e.context}" for e in self.exemplars]
        self.embeddings = self.embedder.encode(texts)
        
        # Save embeddings
        embeddings_file = self.store_path / "adaptive_embeddings.npy"
        np.save(embeddings_file, self.embeddings)
        logger.info(f"Computed and saved embeddings for {len(self.exemplars)} exemplars")
    
    def add_exemplar(self, exemplar: AdaptiveExemplar, success: bool = True):
        """
        Add a new exemplar with adaptive scoring
        
        Args:
            exemplar: Exemplar to add
            success: Whether the exemplar led to success
        """
        # Set timestamps
        current_time = time.time()
        exemplar.created_at = current_time
        exemplar.last_used = current_time
        exemplar.success_count = 1 if success else 0
        exemplar.failure_count = 0 if success else 1
        
        # Compute embedding
        text = f"{exemplar.task} {exemplar.subtask} {exemplar.context}"
        embedding = self.embedder.encode([text])[0]
        exemplar.embedding = embedding.tolist()
        
        # Compute diversity score
        if self.exemplars:
            exemplar.diversity_score = self._compute_diversity(embedding)
        
        # Add to store
        self.exemplars.append(exemplar)
        
        # Update embeddings matrix
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = np.array([embedding])
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        # Save to disk
        self._save_exemplars()
        logger.info(f"Added adaptive exemplar (success={success}), total: {len(self.exemplars)}")
    
    def _compute_diversity(self, new_embedding: np.ndarray) -> float:
        """
        Compute diversity score based on similarity to existing exemplars
        
        Args:
            new_embedding: Embedding of new exemplar
            
        Returns:
            Diversity score (lower similarity = higher diversity)
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return 1.0
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, new_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(new_embedding) + 1e-8
        )
        
        # Diversity = 1 - max_similarity
        max_sim = np.max(similarities)
        diversity = max(0.1, 1.0 - max_sim)
        
        return diversity
    
    def retrieve(
        self,
        query_task: str,
        query_subtask: str,
        query_context: str = ""
    ) -> List[AdaptiveExemplar]:
        """
        Retrieve exemplars using adaptive scoring
        
        Args:
            query_task: Query task
            query_subtask: Query subtask
            query_context: Query context
            
        Returns:
            Retrieved exemplars ranked by adaptive score
        """
        if not self.exemplars:
            return []
        
        # Compute query embedding
        query_text = f"{query_task} {query_subtask} {query_context}"
        query_embedding = self.embedder.encode([query_text])[0]
        
        # Compute semantic similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        # Combine semantic similarity with adaptive score
        combined_scores = []
        for i, exemplar in enumerate(self.exemplars):
            semantic_score = similarities[i]
            adaptive_score = exemplar.adaptive_score
            # Weighted combination: 60% semantic, 40% adaptive
            combined = 0.6 * semantic_score + 0.4 * adaptive_score
            combined_scores.append((i, combined, exemplar))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k
        retrieved = [item[2] for item in combined_scores[:self.top_k]]
        
        # Update last_used for retrieved exemplars
        for exemplar in retrieved:
            exemplar.last_used = time.time()
        
        logger.info(f"Retrieved {len(retrieved)} exemplars using adaptive scoring")
        return retrieved
    
    def update_exemplar_outcome(self, exemplar_index: int, success: bool):
        """
        Update exemplar based on task outcome
        
        Args:
            exemplar_index: Index of exemplar to update
            success: Whether the exemplar led to success
        """
        if 0 <= exemplar_index < len(self.exemplars):
            exemplar = self.exemplars[exemplar_index]
            if success:
                exemplar.success_count += 1
            else:
                exemplar.failure_count += 1
                # Penalize by reducing diversity score temporarily
                exemplar.diversity_score *= 0.9
            
            self._save_exemplars()
            logger.info(f"Updated exemplar {exemplar_index}: success={success}")
    
    def _save_exemplars(self):
        """Save exemplars to disk"""
        exemplar_file = self.store_path / "adaptive_exemplars.json"
        data = [e.to_dict() for e in self.exemplars]
        with open(exemplar_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # Save embeddings
        if self.embeddings is not None and len(self.embeddings) > 0:
            embeddings_file = self.store_path / "adaptive_embeddings.npy"
            np.save(embeddings_file, self.embeddings)
    
    def format_exemplars(self, exemplars: List[AdaptiveExemplar]) -> str:
        """Format exemplars for LLM prompt"""
        if not exemplars:
            return "No relevant exemplars found."
        
        formatted = []
        for i, exemplar in enumerate(exemplars, 1):
            formatted.append(f"Exemplar {i} (Score: {exemplar.adaptive_score:.2f}):")
            formatted.append(f"  Task: {exemplar.task}")
            formatted.append(f"  Subtask: {exemplar.subtask}")
            formatted.append(f"  Actions: {exemplar.actions}")
            formatted.append(f"  Outcome: {exemplar.outcome}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def get_statistics(self) -> Dict:
        """Get store statistics"""
        if not self.exemplars:
            return {
                "total_exemplars": 0,
                "avg_success_rate": 0,
                "avg_adaptive_score": 0
            }
        
        success_rates = [e.success_rate for e in self.exemplars]
        adaptive_scores = [e.adaptive_score for e in self.exemplars]
        
        return {
            "total_exemplars": len(self.exemplars),
            "avg_success_rate": np.mean(success_rates),
            "avg_adaptive_score": np.mean(adaptive_scores),
            "high_performers": sum(1 for e in self.exemplars if e.success_rate > 0.8)
        }
