"""
Vector store for exemplar management
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from loguru import logger

from .embedder import Embedder


@dataclass
class Exemplar:
    """Represents a single exemplar (task trajectory)"""
    task: str
    subtask: str
    context: str
    actions: List[Dict]
    outcome: str
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Exemplar':
        return cls(**data)


class ExemplarStore:
    """Manages storage and retrieval of exemplars"""
    
    def __init__(
        self,
        embedder: Embedder,
        store_path: str = "./exemplars",
        top_k: int = 3
    ):
        """
        Initialize the exemplar store
        
        Args:
            embedder: Embedder instance for computing embeddings
            store_path: Path to store exemplars
            top_k: Number of exemplars to retrieve
        """
        self.embedder = embedder
        self.store_path = Path(store_path)
        self.top_k = top_k
        self.exemplars: List[Exemplar] = []
        self.embeddings: Optional[np.ndarray] = None
        
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._load_exemplars()
    
    def _load_exemplars(self):
        """Load exemplars from disk"""
        exemplar_file = self.store_path / "exemplars.json"
        if exemplar_file.exists():
            with open(exemplar_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.exemplars = [Exemplar.from_dict(e) for e in data]
            logger.info(f"Loaded {len(self.exemplars)} exemplars from disk")
            
            # Load or compute embeddings
            embeddings_file = self.store_path / "embeddings.npy"
            if embeddings_file.exists():
                self.embeddings = np.load(embeddings_file)
                logger.info(f"Loaded embeddings for {len(self.exemplars)} exemplars")
            else:
                self._compute_embeddings()
        else:
            logger.info("No exemplars found on disk, starting with empty store")
    
    def _compute_embeddings(self):
        """Compute embeddings for all exemplars"""
        if not self.exemplars:
            self.embeddings = np.array([])
            return
        
        texts = [f"{e.task} {e.subtask} {e.context}" for e in self.exemplars]
        self.embeddings = self.embedder.encode(texts)
        
        # Save embeddings
        embeddings_file = self.store_path / "embeddings.npy"
        np.save(embeddings_file, self.embeddings)
        logger.info(f"Computed and saved embeddings for {len(self.exemplars)} exemplars")
    
    def add_exemplar(self, exemplar: Exemplar):
        """
        Add a new exemplar to the store
        
        Args:
            exemplar: Exemplar to add
        """
        # Compute embedding for the new exemplar
        text = f"{exemplar.task} {exemplar.subtask} {exemplar.context}"
        embedding = self.embedder.encode_query(text)
        exemplar.embedding = embedding.tolist()
        
        self.exemplars.append(exemplar)
        
        # Update embeddings array
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = np.array([embedding])
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        # Save to disk
        self._save_exemplars()
        logger.info(f"Added exemplar. Total: {len(self.exemplars)}")
    
    def _save_exemplars(self):
        """Save exemplars to disk"""
        exemplar_file = self.store_path / "exemplars.json"
        with open(exemplar_file, 'w', encoding='utf-8') as f:
            json.dump([e.to_dict() for e in self.exemplars], f, indent=2)
        
        # Save embeddings
        if self.embeddings is not None:
            embeddings_file = self.store_path / "embeddings.npy"
            np.save(embeddings_file, self.embeddings)
    
    def retrieve(self, query_task: str, query_subtask: str, query_context: str = "") -> List[Exemplar]:
        """
        Retrieve top-k similar exemplars
        
        Args:
            query_task: Task description
            query_subtask: Subtask description
            query_context: Additional context
            
        Returns:
            List of top-k exemplars
        """
        if not self.exemplars or self.embeddings is None or len(self.embeddings) == 0:
            logger.warning("No exemplars available for retrieval")
            return []
        
        # Encode query
        query_text = f"{query_task} {query_subtask} {query_context}"
        query_embedding = self.embedder.encode_query(query_text)
        
        # Compute similarities
        similarities = self.embedder.similarity(query_embedding, self.embeddings.T)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        # Return top-k exemplars
        retrieved = [self.exemplars[i] for i in top_k_indices]
        logger.info(f"Retrieved {len(retrieved)} exemplars for query")
        
        return retrieved
    
    def format_exemplars(self, exemplars: List[Exemplar]) -> str:
        """
        Format exemplars for use in LLM prompts
        
        Args:
            exemplars: List of exemplars
            
        Returns:
            Formatted string
        """
        if not exemplars:
            return "No exemplars available."
        
        formatted = []
        for i, exemplar in enumerate(exemplars, 1):
            formatted.append(f"Exemplar {i}:")
            formatted.append(f"  Task: {exemplar.task}")
            formatted.append(f"  Subtask: {exemplar.subtask}")
            formatted.append(f"  Actions: {json.dumps(exemplar.actions, indent=4)}")
            formatted.append(f"  Outcome: {exemplar.outcome}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def clear(self):
        """Clear all exemplars"""
        self.exemplars = []
        self.embeddings = None
        self._save_exemplars()
        logger.info("Cleared all exemplars")
