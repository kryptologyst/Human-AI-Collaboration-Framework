"""Evaluation metrics for Human-AI Collaboration Framework."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from scipy.stats import spearmanr, kendalltau
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CollaborationMetrics:
    """Metrics for evaluating human-AI collaboration effectiveness."""
    
    def __init__(self):
        """Initialize collaboration metrics."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.ground_truth = []
        self.confidences = []
        self.feedback_types = []
        self.response_times = []
        self.human_corrections = []
    
    def update(
        self,
        prediction: Union[str, int],
        ground_truth: Union[str, int],
        confidence: float,
        feedback_type: str = "accept",
        response_time: Optional[float] = None,
        human_correction: Optional[Union[str, int]] = None,
    ) -> None:
        """Update metrics with new data point.
        
        Args:
            prediction: Model prediction.
            ground_truth: Ground truth label.
            confidence: Model confidence.
            feedback_type: Type of human feedback.
            response_time: Human response time (optional).
            human_correction: Human correction (optional).
        """
        self.predictions.append(prediction)
        self.ground_truth.append(ground_truth)
        self.confidences.append(confidence)
        self.feedback_types.append(feedback_type)
        
        if response_time is not None:
            self.response_times.append(response_time)
        
        if human_correction is not None:
            self.human_corrections.append(human_correction)
    
    def compute_accuracy(self) -> Dict[str, float]:
        """Compute accuracy metrics.
        
        Returns:
            Dictionary with accuracy metrics.
        """
        if not self.predictions:
            return {"error": "No predictions available"}
        
        # Overall accuracy
        overall_accuracy = accuracy_score(self.ground_truth, self.predictions)
        
        # Accuracy by feedback type
        feedback_accuracies = {}
        for feedback_type in set(self.feedback_types):
            mask = [ft == feedback_type for ft in self.feedback_types]
            if any(mask):
                feedback_accuracies[feedback_type] = accuracy_score(
                    [gt for i, gt in enumerate(self.ground_truth) if mask[i]],
                    [pred for i, pred in enumerate(self.predictions) if mask[i]]
                )
        
        # High confidence accuracy
        high_conf_mask = [conf > 0.8 for conf in self.confidences]
        if any(high_conf_mask):
            high_conf_accuracy = accuracy_score(
                [gt for i, gt in enumerate(self.ground_truth) if high_conf_mask[i]],
                [pred for i, pred in enumerate(self.predictions) if high_conf_mask[i]]
            )
        else:
            high_conf_accuracy = 0.0
        
        return {
            "overall_accuracy": overall_accuracy,
            "feedback_type_accuracies": feedback_accuracies,
            "high_confidence_accuracy": high_conf_accuracy,
            "total_samples": len(self.predictions),
        }
    
    def compute_confidence_metrics(self) -> Dict[str, float]:
        """Compute confidence-related metrics.
        
        Returns:
            Dictionary with confidence metrics.
        """
        if not self.confidences:
            return {"error": "No confidence data available"}
        
        confidences = np.array(self.confidences)
        accuracies = np.array([
            pred == gt for pred, gt in zip(self.predictions, self.ground_truth)
        ])
        
        # Calibration metrics
        calibration_error = self._compute_calibration_error(confidences, accuracies)
        
        # Confidence-accuracy correlation
        correlation = spearmanr(confidences, accuracies.astype(float))[0]
        
        # Overconfidence metrics
        overconfidence = np.mean(confidences - accuracies.astype(float))
        
        return {
            "mean_confidence": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
            "calibration_error": calibration_error,
            "confidence_accuracy_correlation": correlation,
            "overconfidence": overconfidence,
        }
    
    def compute_collaboration_metrics(self) -> Dict[str, Any]:
        """Compute collaboration-specific metrics.
        
        Returns:
            Dictionary with collaboration metrics.
        """
        if not self.predictions:
            return {"error": "No collaboration data available"}
        
        # Feedback distribution
        feedback_distribution = {}
        for feedback_type in set(self.feedback_types):
            feedback_distribution[feedback_type] = self.feedback_types.count(feedback_type)
        
        # Human intervention rate
        intervention_rate = sum(
            1 for ft in self.feedback_types if ft in ["reject", "correction"]
        ) / len(self.feedback_types)
        
        # Response time metrics
        response_time_metrics = {}
        if self.response_times:
            response_times = np.array(self.response_times)
            response_time_metrics = {
                "mean_response_time": float(np.mean(response_times)),
                "median_response_time": float(np.median(response_times)),
                "response_time_std": float(np.std(response_times)),
            }
        
        # Correction effectiveness
        correction_effectiveness = {}
        if self.human_corrections:
            corrections = np.array(self.human_corrections)
            ground_truth_corrected = [
                gt for i, gt in enumerate(self.ground_truth)
                if i < len(corrections)
            ]
            correction_accuracy = accuracy_score(ground_truth_corrected, corrections)
            correction_effectiveness = {
                "correction_accuracy": correction_accuracy,
                "total_corrections": len(corrections),
            }
        
        return {
            "feedback_distribution": feedback_distribution,
            "intervention_rate": intervention_rate,
            "response_time_metrics": response_time_metrics,
            "correction_effectiveness": correction_effectiveness,
        }
    
    def _compute_calibration_error(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute expected calibration error.
        
        Args:
            confidences: Model confidence scores.
            accuracies: Binary accuracy scores.
            n_bins: Number of bins for calibration.
            
        Returns:
            Expected calibration error.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary.
        
        Returns:
            Dictionary with all metrics.
        """
        return {
            "accuracy_metrics": self.compute_accuracy(),
            "confidence_metrics": self.compute_confidence_metrics(),
            "collaboration_metrics": self.compute_collaboration_metrics(),
        }


class ExplanationMetrics:
    """Metrics for evaluating explanation quality."""
    
    def __init__(self):
        """Initialize explanation metrics."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.explanations = []
        self.ground_truth_features = []
        self.explanation_methods = []
    
    def update(
        self,
        explanation: Dict[str, Any],
        ground_truth_features: Optional[List[str]] = None,
        method: str = "unknown",
    ) -> None:
        """Update metrics with new explanation.
        
        Args:
            explanation: Explanation dictionary.
            ground_truth_features: Ground truth important features (optional).
            method: Explanation method name.
        """
        self.explanations.append(explanation)
        self.ground_truth_features.append(ground_truth_features or [])
        self.explanation_methods.append(method)
    
    def compute_faithfulness_metrics(self) -> Dict[str, float]:
        """Compute explanation faithfulness metrics.
        
        Returns:
            Dictionary with faithfulness metrics.
        """
        if not self.explanations:
            return {"error": "No explanations available"}
        
        # This is a simplified implementation
        # In practice, you would need ground truth explanations or human annotations
        
        faithfulness_scores = []
        
        for explanation in self.explanations:
            if "feature_importance" in explanation:
                # Simple faithfulness: higher importance features should be more important
                features = explanation["feature_importance"]
                if len(features) > 1:
                    # Check if importance decreases monotonically
                    importances = [f["importance"] for f in features]
                    sorted_importances = sorted(importances, reverse=True)
                    is_monotonic = importances == sorted_importances
                    faithfulness_scores.append(float(is_monotonic))
        
        return {
            "mean_faithfulness": float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
            "faithfulness_std": float(np.std(faithfulness_scores)) if faithfulness_scores else 0.0,
        }
    
    def compute_stability_metrics(self) -> Dict[str, float]:
        """Compute explanation stability metrics.
        
        Returns:
            Dictionary with stability metrics.
        """
        if len(self.explanations) < 2:
            return {"error": "Need at least 2 explanations for stability analysis"}
        
        # Compare explanations across methods
        stability_scores = []
        
        for i in range(len(self.explanations) - 1):
            for j in range(i + 1, len(self.explanations)):
                if self.explanation_methods[i] != self.explanation_methods[j]:
                    # Compare different methods on same input
                    stability_score = self._compare_explanations(
                        self.explanations[i],
                        self.explanations[j]
                    )
                    stability_scores.append(stability_score)
        
        return {
            "mean_stability": float(np.mean(stability_scores)) if stability_scores else 0.0,
            "stability_std": float(np.std(stability_scores)) if stability_scores else 0.0,
        }
    
    def _compare_explanations(
        self,
        explanation1: Dict[str, Any],
        explanation2: Dict[str, Any],
    ) -> float:
        """Compare two explanations and return similarity score.
        
        Args:
            explanation1: First explanation.
            explanation2: Second explanation.
            
        Returns:
            Similarity score between 0 and 1.
        """
        # Extract features from both explanations
        features1 = self._extract_features(explanation1)
        features2 = self._extract_features(explanation2)
        
        if not features1 or not features2:
            return 0.0
        
        # Compute Jaccard similarity
        set1 = set(features1.keys())
        set2 = set(features2.keys())
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _extract_features(self, explanation: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from explanation.
        
        Args:
            explanation: Explanation dictionary.
            
        Returns:
            Dictionary mapping features to importance scores.
        """
        features = {}
        
        if "feature_importance" in explanation:
            for feat in explanation["feature_importance"]:
                features[feat["feature"]] = feat["importance"]
        
        return features
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive explanation metrics summary.
        
        Returns:
            Dictionary with all explanation metrics.
        """
        return {
            "faithfulness_metrics": self.compute_faithfulness_metrics(),
            "stability_metrics": self.compute_stability_metrics(),
        }


class TrustMetrics:
    """Metrics for evaluating trust and reliability."""
    
    def __init__(self):
        """Initialize trust metrics."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.confidences = []
        self.uncertainties = []
        self.user_ratings = []
        self.explanation_ratings = []
    
    def update(
        self,
        prediction: Union[str, int],
        confidence: float,
        uncertainty: Optional[float] = None,
        user_rating: Optional[float] = None,
        explanation_rating: Optional[float] = None,
    ) -> None:
        """Update trust metrics.
        
        Args:
            prediction: Model prediction.
            confidence: Model confidence.
            uncertainty: Model uncertainty (optional).
            user_rating: User trust rating (optional).
            explanation_rating: User explanation rating (optional).
        """
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        
        if uncertainty is not None:
            self.uncertainties.append(uncertainty)
        
        if user_rating is not None:
            self.user_ratings.append(user_rating)
        
        if explanation_rating is not None:
            self.explanation_ratings.append(explanation_rating)
    
    def compute_trust_metrics(self) -> Dict[str, float]:
        """Compute trust-related metrics.
        
        Returns:
            Dictionary with trust metrics.
        """
        if not self.predictions:
            return {"error": "No trust data available"}
        
        metrics = {}
        
        # Confidence-based trust
        if self.confidences:
            confidences = np.array(self.confidences)
            metrics["mean_confidence"] = float(np.mean(confidences))
            metrics["confidence_consistency"] = float(1.0 - np.std(confidences))
        
        # Uncertainty-based trust
        if self.uncertainties:
            uncertainties = np.array(self.uncertainties)
            metrics["mean_uncertainty"] = float(np.mean(uncertainties))
            metrics["uncertainty_consistency"] = float(1.0 - np.std(uncertainties))
        
        # User-based trust
        if self.user_ratings:
            ratings = np.array(self.user_ratings)
            metrics["mean_user_trust"] = float(np.mean(ratings))
            metrics["user_trust_consistency"] = float(1.0 - np.std(ratings))
        
        # Explanation-based trust
        if self.explanation_ratings:
            ratings = np.array(self.explanation_ratings)
            metrics["mean_explanation_trust"] = float(np.mean(ratings))
            metrics["explanation_trust_consistency"] = float(1.0 - np.std(ratings))
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive trust metrics summary.
        
        Returns:
            Dictionary with all trust metrics.
        """
        return {
            "trust_metrics": self.compute_trust_metrics(),
        }


class FairnessMetrics:
    """Metrics for evaluating fairness and bias."""
    
    def __init__(self):
        """Initialize fairness metrics."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.ground_truth = []
        self.sensitive_attributes = []
        self.confidences = []
    
    def update(
        self,
        prediction: Union[str, int],
        ground_truth: Union[str, int],
        sensitive_attribute: Union[str, int],
        confidence: Optional[float] = None,
    ) -> None:
        """Update fairness metrics.
        
        Args:
            prediction: Model prediction.
            ground_truth: Ground truth label.
            sensitive_attribute: Sensitive attribute value.
            confidence: Model confidence (optional).
        """
        self.predictions.append(prediction)
        self.ground_truth.append(ground_truth)
        self.sensitive_attributes.append(sensitive_attribute)
        
        if confidence is not None:
            self.confidences.append(confidence)
    
    def compute_fairness_metrics(self) -> Dict[str, Any]:
        """Compute fairness metrics.
        
        Returns:
            Dictionary with fairness metrics.
        """
        if not self.predictions:
            return {"error": "No fairness data available"}
        
        # Group metrics by sensitive attribute
        groups = set(self.sensitive_attributes)
        group_metrics = {}
        
        for group in groups:
            group_mask = [sa == group for sa in self.sensitive_attributes]
            group_predictions = [pred for i, pred in enumerate(self.predictions) if group_mask[i]]
            group_ground_truth = [gt for i, gt in enumerate(self.ground_truth) if group_mask[i]]
            
            if group_predictions and group_ground_truth:
                accuracy = accuracy_score(group_ground_truth, group_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    group_ground_truth, group_predictions, average="weighted"
                )
                
                group_metrics[group] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "sample_count": len(group_predictions),
                }
        
        # Compute fairness gaps
        fairness_gaps = {}
        if len(group_metrics) > 1:
            accuracies = [metrics["accuracy"] for metrics in group_metrics.values()]
            fairness_gaps["accuracy_gap"] = float(max(accuracies) - min(accuracies))
            
            f1_scores = [metrics["f1_score"] for metrics in group_metrics.values()]
            fairness_gaps["f1_gap"] = float(max(f1_scores) - min(f1_scores))
        
        return {
            "group_metrics": group_metrics,
            "fairness_gaps": fairness_gaps,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive fairness metrics summary.
        
        Returns:
            Dictionary with all fairness metrics.
        """
        return {
            "fairness_metrics": self.compute_fairness_metrics(),
        }
