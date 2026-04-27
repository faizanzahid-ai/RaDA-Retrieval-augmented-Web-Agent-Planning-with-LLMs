"""
Embedding model for exemplar retrieval
"""

from typing import List, Optional
import numpy as np
from loguru import logger


class Embedder:
    """Handles text embeddings for exemplar retrieval"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the embedder
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except (ImportError, Exception) as e:
            logger.warning(f"sentence-transformers not available ({e}), trying OpenAI embeddings")
            try:
                from openai import OpenAI
                self.openai_client = OpenAI()
                self._use_openai = True
            except Exception:
                logger.warning("OpenAI also unavailable, using random embeddings as fallback")
                self.model = None
                self._use_openai = False
                self._use_random = True
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings
        """
        if hasattr(self, '_use_openai') and self._use_openai:
            return self._encode_openai(texts)
        
        if hasattr(self, '_use_random') and self._use_random:
            logger.warning("Using random embeddings (install sentence-transformers for real embeddings)")
            return np.random.randn(len(texts), 384).astype('float32')
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    
    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        """Encode using OpenAI API as fallback"""
        embeddings = []
        for text in texts:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        
        return np.array(embeddings)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query
        
        Args:
            query: Query text
            
        Returns:
            numpy array of embedding
        """
        return self.encode([query])[0]
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
