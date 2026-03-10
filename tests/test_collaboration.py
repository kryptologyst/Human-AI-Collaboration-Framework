"""Test suite for Human-AI Collaboration Framework."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from human_ai_collaboration import (
    TextClassifier,
    CollaborationModel,
    DataLoader,
    SyntheticDataGenerator,
    SHAPExplainer,
    LIMEExplainer,
    CollaborationMetrics,
    ExplanationMetrics,
    TrustMetrics,
    FairnessMetrics,
    set_seed,
    get_device,
)


class TestTextClassifier:
    """Test cases for TextClassifier."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = TextClassifier(model_name="distilbert-base-uncased", num_classes=5)
        assert model.num_classes == 5
        assert model.device is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = TextClassifier(model_name="distilbert-base-uncased", num_classes=5)
        
        # Mock tokenizer output
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        with patch.object(model, 'encoder') as mock_encoder:
            mock_output = Mock()
            mock_output.last_hidden_state = torch.randn(2, 10, 768)
            mock_encoder.return_value = mock_output
            
            logits = model.forward(input_ids, attention_mask)
            assert logits.shape == (2, 5)
    
    def test_predict(self):
        """Test prediction method."""
        model = TextClassifier(model_name="distilbert-base-uncased", num_classes=5)
        
        with patch.object(model, 'forward') as mock_forward:
            mock_forward.return_value = torch.randn(2, 5)
            
            texts = ["Sample text 1", "Sample text 2"]
            predictions, probabilities = model.predict(texts)
            
            assert len(predictions) == 2
            assert len(probabilities) == 2
            assert predictions.shape == (2,)
            assert probabilities.shape == (2, 5)


class TestCollaborationModel:
    """Test cases for CollaborationModel."""
    
    def test_initialization(self):
        """Test collaboration model initialization."""
        base_model = TextClassifier(model_name="distilbert-base-uncased", num_classes=5)
        collab_model = CollaborationModel(base_model=base_model)
        
        assert collab_model.confidence_threshold == 0.8
        assert collab_model.feedback_weight == 0.1
        assert len(collab_model.feedback_history) == 0
    
    def test_suggest_label(self):
        """Test label suggestion."""
        base_model = TextClassifier(model_name="distilbert-base-uncased", num_classes=5)
        collab_model = CollaborationModel(base_model=base_model)
        
        with patch.object(base_model, 'predict') as mock_predict:
            mock_predict.return_value = (np.array([2]), np.array([[0.1, 0.1, 0.8, 0.0, 0.0]]))
            
            suggested_label, confidence = collab_model.suggest_label(
                "Sample text", ["A", "B", "C", "D", "E"]
            )
            
            assert suggested_label == "C"
            assert confidence == 0.8
    
    def test_process_feedback(self):
        """Test feedback processing."""
        base_model = TextClassifier(model_name="distilbert-base-uncased", num_classes=5)
        collab_model = CollaborationModel(base_model=base_model)
        
        result = collab_model.process_feedback(
            text="Sample text",
            ai_suggestion="A",
            human_label="B",
            confidence=0.7,
            feedback_type="correction"
        )
        
        assert result["feedback_processed"] is True
        assert result["is_correct"] is False
        assert len(collab_model.feedback_history) == 1
    
    def test_should_request_feedback(self):
        """Test feedback request logic."""
        base_model = TextClassifier(model_name="distilbert-base-uncased", num_classes=5)
        collab_model = CollaborationModel(base_model=base_model, confidence_threshold=0.8)
        
        assert collab_model.should_request_human_feedback(0.7) is True
        assert collab_model.should_request_human_feedback(0.9) is False


class TestDataLoader:
    """Test cases for DataLoader."""
    
    def test_initialization(self):
        """Test data loader initialization."""
        loader = DataLoader()
        assert loader.max_length == 512
        assert loader.random_seed == 42
    
    def test_load_custom_data(self):
        """Test custom data loading."""
        loader = DataLoader()
        
        texts = ["Text 1", "Text 2", "Text 3"]
        labels = ["A", "B", "A"]
        label_names = ["A", "B"]
        
        loaded_texts, loaded_labels, loaded_names = loader.load_custom_data(
            texts, labels, label_names
        )
        
        assert len(loaded_texts) == 3
        assert len(loaded_labels) == 3
        assert len(loaded_names) == 2
    
    def test_create_splits(self):
        """Test data splitting."""
        loader = DataLoader()
        
        texts = ["Text"] * 100
        labels = np.random.randint(0, 5, 100)
        
        train, val, test = loader.create_splits(texts, labels)
        
        assert len(train[0]) + len(val[0]) + len(test[0]) == 100
        assert len(train[1]) + len(val[1]) + len(test[1]) == 100


class TestSyntheticDataGenerator:
    """Test cases for SyntheticDataGenerator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = SyntheticDataGenerator()
        assert generator.random_seed == 42
    
    def test_generate_data(self):
        """Test synthetic data generation."""
        generator = SyntheticDataGenerator()
        
        texts, labels, label_names = generator.generate_text_classification_data(
            n_samples=50, n_classes=3
        )
        
        assert len(texts) == 50
        assert len(labels) == 50
        assert len(label_names) == 3
        assert max(labels) < 3


class TestMetrics:
    """Test cases for metrics classes."""
    
    def test_collaboration_metrics(self):
        """Test collaboration metrics."""
        metrics = CollaborationMetrics()
        
        # Add some test data
        metrics.update("A", "A", 0.9, "accept")
        metrics.update("B", "A", 0.7, "reject")
        metrics.update("C", "C", 0.8, "accept")
        
        accuracy_metrics = metrics.compute_accuracy()
        assert accuracy_metrics["overall_accuracy"] == 2/3
        
        confidence_metrics = metrics.compute_confidence_metrics()
        assert "mean_confidence" in confidence_metrics
    
    def test_explanation_metrics(self):
        """Test explanation metrics."""
        metrics = ExplanationMetrics()
        
        # Add test explanations
        explanation1 = {
            "feature_importance": [
                {"feature": "word1", "importance": 0.5},
                {"feature": "word2", "importance": 0.3}
            ]
        }
        
        explanation2 = {
            "feature_importance": [
                {"feature": "word1", "importance": 0.4},
                {"feature": "word2", "importance": 0.4}
            ]
        }
        
        metrics.update(explanation1, method="SHAP")
        metrics.update(explanation2, method="LIME")
        
        summary = metrics.get_summary()
        assert "faithfulness_metrics" in summary
        assert "stability_metrics" in summary
    
    def test_trust_metrics(self):
        """Test trust metrics."""
        metrics = TrustMetrics()
        
        metrics.update("A", 0.8, uncertainty=0.2, user_rating=4.0)
        metrics.update("B", 0.9, uncertainty=0.1, user_rating=5.0)
        
        trust_metrics = metrics.compute_trust_metrics()
        assert "mean_confidence" in trust_metrics
        assert "mean_user_trust" in trust_metrics
    
    def test_fairness_metrics(self):
        """Test fairness metrics."""
        metrics = FairnessMetrics()
        
        # Add test data with sensitive attributes
        metrics.update("A", "A", "group1", confidence=0.8)
        metrics.update("B", "A", "group1", confidence=0.7)
        metrics.update("A", "A", "group2", confidence=0.9)
        metrics.update("B", "B", "group2", confidence=0.8)
        
        fairness_metrics = metrics.compute_fairness_metrics()
        assert "group_metrics" in fairness_metrics
        assert "fairness_gaps" in fairness_metrics


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(123)
        # This is hard to test directly, but we can verify it doesn't raise an error
        assert True
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_collaboration(self):
        """Test end-to-end collaboration workflow."""
        # Initialize components
        base_model = TextClassifier(model_name="distilbert-base-uncased", num_classes=3)
        collab_model = CollaborationModel(base_model=base_model)
        metrics = CollaborationMetrics()
        
        # Generate synthetic data
        generator = SyntheticDataGenerator()
        texts, labels, label_names = generator.generate_text_classification_data(
            n_samples=10, n_classes=3
        )
        
        # Simulate collaboration
        for i, text in enumerate(texts[:5]):
            with patch.object(base_model, 'predict') as mock_predict:
                mock_predict.return_value = (
                    np.array([labels[i]]),
                    np.array([[0.1, 0.8, 0.1]])
                )
                
                suggested_label, confidence = collab_model.suggest_label(text, label_names)
                
                # Process feedback
                result = collab_model.process_feedback(
                    text=text,
                    ai_suggestion=suggested_label,
                    human_label=label_names[labels[i]],
                    confidence=confidence
                )
                
                # Update metrics
                metrics.update(
                    suggested_label,
                    label_names[labels[i]],
                    confidence,
                    "accept" if result["is_correct"] else "correction"
                )
        
        # Verify results
        assert len(collab_model.feedback_history) == 5
        assert metrics.compute_accuracy()["total_samples"] == 5


if __name__ == "__main__":
    pytest.main([__file__])
