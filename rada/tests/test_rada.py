"""Tests for RaDA implementation"""

import sys
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Pre-register a mock module for sentence_transformers so patch() works
# even when the real package is not installed
_mock_st = MagicMock()
sys.modules.setdefault('sentence_transformers', _mock_st)

from rada.retrieval.embedder import Embedder
from rada.retrieval.store import ExemplarStore, Exemplar
from rada.env.dom_parser import DOMParser
from rada.evaluation.metrics import TaskSuccessRate, ActionEfficiency, SubtaskCompletionRate


class TestEmbedder:
    """Test Embedder class"""
    
    def test_initialization(self):
        """Test embedder initialization"""
        with patch('sentence_transformers.SentenceTransformer'):
            embedder = Embedder(model_name="test-model")
            assert embedder.model_name == "test-model"
    
    def test_similarity(self):
        """Test cosine similarity calculation"""
        import numpy as np
        with patch('sentence_transformers.SentenceTransformer'):
            embedder = Embedder()
            
            # Identical vectors should have similarity 1.0
            vec1 = np.array([1, 0, 0])
            vec2 = np.array([1, 0, 0])
            similarity = embedder.similarity(vec1, vec2)
            assert abs(similarity - 1.0) < 0.001
            
            # Orthogonal vectors should have similarity 0.0
            vec3 = np.array([0, 1, 0])
            similarity = embedder.similarity(vec1, vec3)
            assert abs(similarity) < 0.001


class TestExemplarStore:
    """Test ExemplarStore class"""
    
    @pytest.fixture
    def setup_store(self, tmp_path):
        """Setup test store"""
        # Use random embedding fallback instead of patching sentence_transformers
        import numpy as np
        embedder = Embedder()
        embedder.model = None
        embedder._use_random = True
        # Override encode to return real numpy arrays
        embedder.encode = lambda texts, batch_size=32: np.random.randn(len(texts), 384).astype('float32')
        embedder.encode_query = lambda text: np.random.randn(384).astype('float32')
        store = ExemplarStore(embedder=embedder, store_path=str(tmp_path))
        return store
    
    def test_add_exemplar(self, setup_store):
        """Test adding exemplar"""
        store = setup_store
        exemplar = Exemplar(
            task="Test task",
            subtask="Test subtask",
            context="Test context",
            actions=[{"action": "CLICK"}],
            outcome="success"
        )
        
        initial_count = len(store.exemplars)
        store.add_exemplar(exemplar)
        
        assert len(store.exemplars) == initial_count + 1
    
    def test_retrieve_empty(self, setup_store):
        """Test retrieval from empty store"""
        store = setup_store
        exemplars = store.retrieve("task", "subtask")
        assert len(exemplars) == 0
    
    def test_format_exemplars(self, setup_store):
        """Test exemplar formatting"""
        store = setup_store
        exemplars = []
        formatted = store.format_exemplars(exemplars)
        assert "No exemplars available" in formatted


class TestDOMParser:
    """Test DOMParser class"""
    
    def test_extract_interactive_elements(self):
        """Test extracting interactive elements"""
        parser = DOMParser()
        html = """
        <html>
            <body>
                <a href="/link">Click here</a>
                <button id="btn1">Submit</button>
                <input type="text" placeholder="Enter name" />
            </body>
        </html>
        """
        
        elements = parser.parse(html)
        assert len(elements) == 3
        
        # Check element types
        tags = [elem['tag'] for elem in elements]
        assert 'a' in tags
        assert 'button' in tags
        assert 'input' in tags
    
    def test_format_elements(self):
        """Test element formatting"""
        parser = DOMParser()
        elements = [
            {"element_id": "elem_0000", "tag": "button", "text": "Submit", "action_type": "CLICK", "xpath": "//button", "attributes": {"id": "btn1", "name": "", "type": "submit", "class": []}},
            {"element_id": "elem_0001", "tag": "input", "text": "", "action_type": "TYPE", "xpath": "//input", "attributes": {"id": "", "name": "", "type": "text", "class": []}}
        ]
        
        formatted = parser.format_for_llm(elements)
        assert "Submit" in formatted
        assert "button" in formatted


class TestMetrics:
    """Test evaluation metrics"""
    
    def test_task_success_rate(self):
        """Test task success rate calculation"""
        metrics = TaskSuccessRate()
        
        metrics.update({"success": True})
        metrics.update({"success": False})
        metrics.update({"success": True})
        
        assert metrics.total_tasks == 3
        assert metrics.successful_tasks == 2
        assert abs(metrics.get_success_rate() - 0.667) < 0.01
    
    def test_action_efficiency(self):
        """Test action efficiency calculation"""
        metrics = ActionEfficiency()
        
        metrics.update({"total_actions": 5})
        metrics.update({"total_actions": 10})
        metrics.update({"total_actions": 15})
        
        assert metrics.total_tasks == 3
        assert metrics.total_actions == 30
        assert metrics.get_average_actions() == 10.0
    
    def test_subtask_completion_rate(self):
        """Test subtask completion rate"""
        metrics = SubtaskCompletionRate()
        
        subtask_results = [
            {"success": True},
            {"success": True},
            {"success": False},
            {"success": True}
        ]
        
        metrics.update(subtask_results)
        
        assert metrics.total_subtasks == 4
        assert metrics.completed_subtasks == 3
        assert abs(metrics.get_completion_rate() - 0.75) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
