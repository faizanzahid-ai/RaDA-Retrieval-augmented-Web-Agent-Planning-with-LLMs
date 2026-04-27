"""
FAISS-based Retriever with Embeddings

Advanced retrieval system using FAISS for efficient similarity search
over trajectory embeddings
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Install with: pip install faiss-cpu")


class EmbeddingModel:
    """Embedding model for trajectory encoding"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available")
            self.model = None
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        if self.model is None:
            # Fallback: random embeddings (for testing only)
            logger.warning("Using random embeddings (install sentence-transformers)")
            return np.random.randn(len(texts), 384).astype('float32')
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.astype('float32')
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text"""
        return self.encode([text])[0]


class TrajectoryRetriever:
    """
    FAISS-based retriever for trajectory memory
    
    Features:
    - Efficient similarity search with FAISS
    - Embedding-based retrieval
    - Trajectory storage and loading
    - Top-k retrieval
    """
    
    def __init__(
        self,
        trajectories_path: str = "./data/trajectories.json",
        index_path: str = "./data/faiss_index.bin",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3
    ):
        """
        Initialize retriever
        
        Args:
            trajectories_path: Path to trajectories JSON
            index_path: Path to save FAISS index
            embedding_model: Embedding model name
            top_k: Number of trajectories to retrieve
        """
        self.trajectories_path = Path(trajectories_path)
        self.index_path = Path(index_path)
        self.top_k = top_k
        
        # Initialize components
        self.embedding_model = EmbeddingModel(model_name=embedding_model)
        self.trajectories = []
        self.index = None
        self.embedding_dim = 384  # Default for MiniLM
        
        # Load or initialize
        self._load_trajectories()
        self._build_or_load_index()
    
    def _load_trajectories(self):
        """Load trajectories from JSON"""
        if self.trajectories_path.exists():
            with open(self.trajectories_path, 'r') as f:
                data = json.load(f)
                self.trajectories = data.get('trajectories', [])
            logger.info(f"Loaded {len(self.trajectories)} trajectories")
        else:
            logger.warning(f"Trajectories file not found: {self.trajectories_path}")
            self.trajectories = []
    
    def _build_or_load_index(self):
        """Build FAISS index or load existing one"""
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available, retrieval disabled")
            return
        
        if self.index_path.exists():
            self._load_index()
        else:
            self._build_index()
    
    def _build_index(self):
        """Build FAISS index from trajectories"""
        if not self.trajectories:
            logger.warning("No trajectories to build index")
            return
        
        # Create texts for embedding
        texts = [
            f"{t['task']} {t['subtask']} {t.get('outcome', '')}"
            for t in self.trajectories
        ]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        self.embedding_dim = embeddings.shape[1]
        
        # Build FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)
        
        # Save index
        self._save_index()
        
        logger.info(f"Built FAISS index with {len(self.trajectories)} trajectories")
    
    def _load_index(self):
        """Load FAISS index from disk"""
        if not FAISS_AVAILABLE:
            return
        
        self.index = faiss.read_index(str(self.index_path))
        self.embedding_dim = self.index.d
        logger.info(f"Loaded FAISS index (dim={self.embedding_dim})")
    
    def _save_index(self):
        """Save FAISS index to disk"""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"Saved FAISS index to {self.index_path}")
    
    def retrieve(
        self,
        query_task: str,
        query_subtask: str,
        query_context: str = "",
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar trajectories
        
        Args:
            query_task: Current task
            query_subtask: Current subtask
            query_context: Additional context
            top_k: Number of results (overrides default)
            
        Returns:
            List of similar trajectories
        """
        if not self.trajectories or self.index is None:
            logger.warning("No trajectories or index available")
            return []
        
        k = top_k or self.top_k
        k = min(k, len(self.trajectories))
        
        # Encode query
        query_text = f"{query_task} {query_subtask} {query_context}"
        query_embedding = self.embedding_model.encode_single(query_text)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return matching trajectories
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.trajectories):
                traj = self.trajectories[idx].copy()
                traj['similarity_score'] = float(1.0 / (1.0 + dist))  # Convert distance to similarity
                results.append(traj)
        
        logger.debug(f"Retrieved {len(results)} trajectories for query")
        return results
    
    def add_trajectory(self, trajectory: Dict[str, Any]):
        """
        Add new trajectory to memory
        
        Args:
            trajectory: New trajectory data
        """
        # Add ID if not present
        if 'id' not in trajectory:
            trajectory['id'] = f"traj_{len(self.trajectories) + 1:04d}"
        
        self.trajectories.append(trajectory)
        
        # Rebuild index
        self._build_index()
        
        # Save trajectories
        self._save_trajectories()
        
        logger.info(f"Added trajectory: {trajectory['id']}")
    
    def _save_trajectories(self):
        """Save trajectories to JSON"""
        self.trajectories_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'trajectories': self.trajectories,
            'metadata': {
                'version': '1.0',
                'total_trajectories': len(self.trajectories)
            }
        }
        
        with open(self.trajectories_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def format_for_prompt(self, trajectories: List[Dict[str, Any]]) -> str:
        """
        Format trajectories for LLM prompt
        
        Args:
            trajectories: Retrieved trajectories
            
        Returns:
            Formatted string
        """
        if not trajectories:
            return "No similar examples available."
        
        formatted = []
        formatted.append("Similar task examples:\n")
        
        for i, traj in enumerate(trajectories, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"  Task: {traj['task']}")
            formatted.append(f"  Subtask: {traj['subtask']}")
            formatted.append(f"  Actions: {len(traj.get('actions', []))}")
            
            # Format actions
            for j, action in enumerate(traj.get('actions', []), 1):
                action_type = action.get('action_type', 'UNKNOWN')
                formatted.append(f"    {j}. {action_type}")
            
            formatted.append(f"  Outcome: {traj.get('outcome', 'unknown')}")
            formatted.append(f"  Similarity: {traj.get('similarity_score', 0):.2f}")
            formatted.append("")
        
        return '\n'.join(formatted)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            'total_trajectories': len(self.trajectories),
            'embedding_dim': self.embedding_dim,
            'index_built': self.index is not None,
            'top_k': self.top_k
        }
