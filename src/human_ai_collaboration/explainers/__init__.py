"""Explanation methods for Human-AI Collaboration Framework."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    DeepLift,
    Saliency,
    InputXGradient,
)
from captum.attr._utils.attribution import Attribution
from lime.lime_text import LimeTextExplainer
import shap
from transformers import AutoTokenizer

from ..models import TextClassifier
from ..utils import get_device

logger = logging.getLogger(__name__)


class BaseExplainer:
    """Base class for explanation methods."""
    
    def __init__(self, model: TextClassifier):
        """Initialize explainer.
        
        Args:
            model: Text classifier model to explain.
        """
        self.model = model
        self.device = get_device()
        self.model.eval()
    
    def explain(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate explanation for a text.
        
        Args:
            text: Input text to explain.
            **kwargs: Additional explanation parameters.
            
        Returns:
            Dictionary with explanation results.
        """
        raise NotImplementedError


class SHAPExplainer(BaseExplainer):
    """SHAP-based explainer for text classification."""
    
    def __init__(
        self,
        model: TextClassifier,
        background_samples: int = 100,
        max_features: int = 1000,
    ):
        """Initialize SHAP explainer.
        
        Args:
            model: Text classifier model.
            background_samples: Number of background samples for SHAP.
            max_features: Maximum number of features to explain.
        """
        super().__init__(model)
        self.background_samples = background_samples
        self.max_features = max_features
        
        # Initialize SHAP explainer
        self.explainer = shap.Explainer(self._model_predict, self._get_background_data())
    
    def _model_predict(self, texts: List[str]) -> np.ndarray:
        """Model prediction function for SHAP.
        
        Args:
            texts: List of input texts.
            
        Returns:
            Prediction probabilities.
        """
        _, probabilities = self.model.predict(texts)
        return probabilities
    
    def _get_background_data(self) -> List[str]:
        """Get background data for SHAP explainer.
        
        Returns:
            List of background texts.
        """
        # Generate synthetic background data
        background_texts = [
            "This is a sample text for background data.",
            "Another example text for SHAP background.",
            "Sample text with different content.",
        ] * (self.background_samples // 3)
        
        return background_texts[:self.background_samples]
    
    def explain(self, text: str, class_idx: Optional[int] = None) -> Dict[str, Any]:
        """Generate SHAP explanation.
        
        Args:
            text: Input text to explain.
            class_idx: Class index to explain (None for predicted class).
            
        Returns:
            Dictionary with SHAP explanation.
        """
        # Get prediction
        predictions, probabilities = self.model.predict([text])
        predicted_class = predictions[0]
        
        if class_idx is None:
            class_idx = predicted_class
        
        # Generate SHAP values
        shap_values = self.explainer([text])
        
        # Extract feature importance
        feature_importance = shap_values.values[0][:, class_idx]
        feature_names = shap_values.data[0]
        
        # Get top features
        top_indices = np.argsort(np.abs(feature_importance))[-self.max_features:][::-1]
        top_features = [
            {"feature": feature_names[i], "importance": float(feature_importance[i])}
            for i in top_indices
        ]
        
        return {
            "explanation_type": "SHAP",
            "predicted_class": int(predicted_class),
            "explained_class": int(class_idx),
            "confidence": float(np.max(probabilities[0])),
            "feature_importance": top_features,
            "shap_values": feature_importance.tolist(),
        }


class LIMEExplainer(BaseExplainer):
    """LIME-based explainer for text classification."""
    
    def __init__(
        self,
        model: TextClassifier,
        num_features: int = 10,
        num_samples: int = 5000,
    ):
        """Initialize LIME explainer.
        
        Args:
            model: Text classifier model.
            num_features: Number of top features to return.
            num_samples: Number of samples for LIME.
        """
        super().__init__(model)
        self.num_features = num_features
        self.num_samples = num_samples
        
        # Initialize LIME explainer
        self.explainer = LimeTextExplainer(class_names=self._get_class_names())
    
    def _get_class_names(self) -> List[str]:
        """Get class names for LIME.
        
        Returns:
            List of class names.
        """
        # Default class names - should be updated based on actual labels
        return [f"Class_{i}" for i in range(self.model.num_classes)]
    
    def _model_predict_proba(self, texts: List[str]) -> np.ndarray:
        """Model prediction function for LIME.
        
        Args:
            texts: List of input texts.
            
        Returns:
            Prediction probabilities.
        """
        _, probabilities = self.model.predict(texts)
        return probabilities
    
    def explain(self, text: str, class_idx: Optional[int] = None) -> Dict[str, Any]:
        """Generate LIME explanation.
        
        Args:
            text: Input text to explain.
            class_idx: Class index to explain (None for predicted class).
            
        Returns:
            Dictionary with LIME explanation.
        """
        # Get prediction
        predictions, probabilities = self.model.predict([text])
        predicted_class = predictions[0]
        
        if class_idx is None:
            class_idx = predicted_class
        
        # Generate LIME explanation
        explanation = self.explainer.explain_instance(
            text,
            self._model_predict_proba,
            num_features=self.num_features,
            num_samples=self.num_samples,
        )
        
        # Extract feature importance
        feature_importance = explanation.as_list()
        
        return {
            "explanation_type": "LIME",
            "predicted_class": int(predicted_class),
            "explained_class": int(class_idx),
            "confidence": float(np.max(probabilities[0])),
            "feature_importance": [
                {"feature": feat, "importance": float(imp)}
                for feat, imp in feature_importance
            ],
            "explanation_score": float(explanation.score),
        }


class AttentionExplainer(BaseExplainer):
    """Attention-based explainer for transformer models."""
    
    def __init__(self, model: TextClassifier):
        """Initialize attention explainer.
        
        Args:
            model: Text classifier model.
        """
        super().__init__(model)
        
        if not hasattr(model.encoder, "encoder"):
            raise ValueError("Model must have encoder layers for attention visualization")
    
    def explain(self, text: str, layer_idx: int = -1) -> Dict[str, Any]:
        """Generate attention explanation.
        
        Args:
            text: Input text to explain.
            layer_idx: Layer index to extract attention from (-1 for last layer).
            
        Returns:
            Dictionary with attention explanation.
        """
        # Tokenize text
        inputs = self.model.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        
        # Extract attention weights
        attentions = outputs.attentions[layer_idx]  # Shape: (batch, heads, seq, seq)
        
        # Average across heads
        attention_weights = attentions.mean(dim=1).squeeze(0)  # Shape: (seq, seq)
        
        # Get tokens
        tokens = self.model.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Get prediction
        predictions, probabilities = self.model.predict([text])
        predicted_class = predictions[0]
        
        # Calculate attention to [CLS] token
        cls_attention = attention_weights[0, :].cpu().numpy()
        
        # Create attention map
        attention_map = [
            {"token": token, "attention": float(attn)}
            for token, attn in zip(tokens, cls_attention)
        ]
        
        return {
            "explanation_type": "Attention",
            "predicted_class": int(predicted_class),
            "confidence": float(np.max(probabilities[0])),
            "attention_map": attention_map,
            "layer_index": layer_idx,
            "tokens": tokens,
        }


class IntegratedGradientsExplainer(BaseExplainer):
    """Integrated Gradients explainer for text classification."""
    
    def __init__(self, model: TextClassifier):
        """Initialize Integrated Gradients explainer.
        
        Args:
            model: Text classifier model.
        """
        super().__init__(model)
        self.ig = IntegratedGradients(self.model)
    
    def explain(self, text: str, class_idx: Optional[int] = None) -> Dict[str, Any]:
        """Generate Integrated Gradients explanation.
        
        Args:
            text: Input text to explain.
            class_idx: Class index to explain (None for predicted class).
            
        Returns:
            Dictionary with Integrated Gradients explanation.
        """
        # Tokenize text
        inputs = self.model.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Get prediction
        predictions, probabilities = self.model.predict([text])
        predicted_class = predictions[0]
        
        if class_idx is None:
            class_idx = predicted_class
        
        # Create baseline (zero input)
        baseline_ids = torch.zeros_like(input_ids)
        baseline_mask = torch.zeros_like(attention_mask)
        
        # Compute integrated gradients
        attributions = self.ig.attribute(
            inputs=(input_ids, attention_mask),
            baselines=(baseline_ids, baseline_mask),
            target=class_idx,
            n_steps=50,
        )
        
        # Extract token attributions
        token_attributions = attributions[0].squeeze(0).cpu().numpy()
        tokens = self.model.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Create attribution map
        attribution_map = [
            {"token": token, "attribution": float(attr)}
            for token, attr in zip(tokens, token_attributions)
        ]
        
        # Sort by absolute attribution
        attribution_map.sort(key=lambda x: abs(x["attribution"]), reverse=True)
        
        return {
            "explanation_type": "IntegratedGradients",
            "predicted_class": int(predicted_class),
            "explained_class": int(class_idx),
            "confidence": float(np.max(probabilities[0])),
            "attribution_map": attribution_map,
            "total_attribution": float(np.sum(token_attributions)),
        }


class ExplanationEnsemble:
    """Ensemble of explanation methods for robust explanations."""
    
    def __init__(self, model: TextClassifier, methods: Optional[List[str]] = None):
        """Initialize explanation ensemble.
        
        Args:
            model: Text classifier model.
            methods: List of explanation methods to use.
        """
        self.model = model
        self.methods = methods or ["SHAP", "LIME", "Attention", "IntegratedGradients"]
        
        # Initialize explainers
        self.explainers = {}
        
        if "SHAP" in self.methods:
            self.explainers["SHAP"] = SHAPExplainer(model)
        
        if "LIME" in self.methods:
            self.explainers["LIME"] = LIMEExplainer(model)
        
        if "Attention" in self.methods:
            try:
                self.explainers["Attention"] = AttentionExplainer(model)
            except ValueError:
                logger.warning("Attention explainer not available for this model")
        
        if "IntegratedGradients" in self.methods:
            self.explainers["IntegratedGradients"] = IntegratedGradientsExplainer(model)
    
    def explain(self, text: str, class_idx: Optional[int] = None) -> Dict[str, Any]:
        """Generate ensemble explanation.
        
        Args:
            text: Input text to explain.
            class_idx: Class index to explain (None for predicted class).
            
        Returns:
            Dictionary with ensemble explanation results.
        """
        explanations = {}
        
        # Get individual explanations
        for method_name, explainer in self.explainers.items():
            try:
                explanations[method_name] = explainer.explain(text, class_idx)
            except Exception as e:
                logger.warning(f"Failed to generate {method_name} explanation: {e}")
                explanations[method_name] = {"error": str(e)}
        
        # Get prediction
        predictions, probabilities = self.model.predict([text])
        predicted_class = predictions[0]
        
        return {
            "text": text,
            "predicted_class": int(predicted_class),
            "confidence": float(np.max(probabilities[0])),
            "explanations": explanations,
            "available_methods": list(self.explainers.keys()),
        }
    
    def get_consensus_features(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get consensus features across explanation methods.
        
        Args:
            text: Input text to explain.
            top_k: Number of top features to return.
            
        Returns:
            List of consensus features.
        """
        explanations = self.explain(text)
        
        # Collect features from all methods
        all_features = {}
        
        for method_name, explanation in explanations["explanations"].items():
            if "error" in explanation:
                continue
            
            if method_name == "SHAP" and "feature_importance" in explanation:
                for feat in explanation["feature_importance"]:
                    feature = feat["feature"]
                    importance = feat["importance"]
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(importance)
            
            elif method_name == "LIME" and "feature_importance" in explanation:
                for feat in explanation["feature_importance"]:
                    feature = feat["feature"]
                    importance = feat["importance"]
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(importance)
        
        # Calculate consensus scores
        consensus_features = []
        for feature, scores in all_features.items():
            if len(scores) > 1:  # Only features explained by multiple methods
                consensus_score = np.mean(scores)
                consensus_features.append({
                    "feature": feature,
                    "consensus_score": consensus_score,
                    "method_count": len(scores),
                    "individual_scores": scores,
                })
        
        # Sort by consensus score
        consensus_features.sort(key=lambda x: abs(x["consensus_score"]), reverse=True)
        
        return consensus_features[:top_k]
