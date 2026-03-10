"""Visualization utilities for Human-AI Collaboration Framework."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

logger = logging.getLogger(__name__)


class ExplanationVisualizer:
    """Visualizer for explanation results."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialize visualizer.
        
        Args:
            style: Matplotlib style to use.
        """
        plt.style.use(style)
        self.colors = px.colors.qualitative.Set3
    
    def plot_feature_importance(
        self,
        explanation: Dict[str, Any],
        method: str = "SHAP",
        top_k: int = 10,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot feature importance from explanation.
        
        Args:
            explanation: Explanation dictionary.
            method: Explanation method name.
            top_k: Number of top features to show.
            figsize: Figure size.
            save_path: Path to save figure (optional).
            
        Returns:
            Matplotlib figure.
        """
        if "feature_importance" not in explanation:
            raise ValueError("Explanation must contain 'feature_importance'")
        
        features = explanation["feature_importance"][:top_k]
        feature_names = [f["feature"] for f in features]
        importances = [f["importance"] for f in features]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, importances, color=self.colors[0])
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Feature Importance - {method}")
        ax.grid(axis="x", alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(
                importance + 0.01 * max(importances),
                bar.get_y() + bar.get_height() / 2,
                f"{importance:.3f}",
                va="center",
                ha="left",
            )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_attention_heatmap(
        self,
        explanation: Dict[str, Any],
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot attention heatmap.
        
        Args:
            explanation: Explanation dictionary with attention data.
            figsize: Figure size.
            save_path: Path to save figure (optional).
            
        Returns:
            Matplotlib figure.
        """
        if "attention_map" not in explanation:
            raise ValueError("Explanation must contain 'attention_map'")
        
        attention_map = explanation["attention_map"]
        tokens = [item["token"] for item in attention_map]
        attentions = [item["attention"] for item in attention_map]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        attention_matrix = np.array(attentions).reshape(1, -1)
        im = ax.imshow(attention_matrix, cmap="Blues", aspect="auto")
        
        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticks([])
        ax.set_title("Attention Heatmap")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight")
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_explanation_comparison(
        self,
        explanations: Dict[str, Dict[str, Any]],
        top_k: int = 10,
        figsize: Tuple[int, int] = (15, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot comparison of different explanation methods.
        
        Args:
            explanations: Dictionary mapping method names to explanations.
            top_k: Number of top features to show.
            figsize: Figure size.
            save_path: Path to save figure (optional).
            
        Returns:
            Matplotlib figure.
        """
        n_methods = len(explanations)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        
        if n_methods == 1:
            axes = [axes]
        
        for i, (method, explanation) in enumerate(explanations.items()):
            if "feature_importance" not in explanation:
                continue
            
            features = explanation["feature_importance"][:top_k]
            feature_names = [f["feature"] for f in features]
            importances = [f["importance"] for f in features]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(feature_names))
            axes[i].barh(y_pos, importances, color=self.colors[i % len(self.colors)])
            
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(feature_names)
            axes[i].set_xlabel("Importance Score")
            axes[i].set_title(f"{method}")
            axes[i].grid(axis="x", alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig


class CollaborationVisualizer:
    """Visualizer for collaboration metrics."""
    
    def __init__(self):
        """Initialize collaboration visualizer."""
        self.colors = px.colors.qualitative.Set2
    
    def plot_accuracy_over_time(
        self,
        metrics_history: List[Dict[str, Any]],
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot accuracy over time.
        
        Args:
            metrics_history: List of metrics dictionaries over time.
            figsize: Figure size.
            save_path: Path to save figure (optional).
            
        Returns:
            Matplotlib figure.
        """
        timestamps = range(len(metrics_history))
        accuracies = [m.get("overall_accuracy", 0) for m in metrics_history]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(timestamps, accuracies, marker="o", color=self.colors[0], linewidth=2)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Over Time")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_confidence_distribution(
        self,
        confidences: List[float],
        accuracies: List[bool],
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot confidence distribution by accuracy.
        
        Args:
            confidences: List of confidence scores.
            accuracies: List of accuracy values (True/False).
            figsize: Figure size.
            save_path: Path to save figure (optional).
            
        Returns:
            Matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Confidence distribution
        ax1.hist(confidences, bins=20, alpha=0.7, color=self.colors[0])
        ax1.set_xlabel("Confidence Score")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Confidence Distribution")
        ax1.grid(True, alpha=0.3)
        
        # Confidence vs Accuracy
        correct_confidences = [c for c, a in zip(confidences, accuracies) if a]
        incorrect_confidences = [c for c, a in zip(confidences, accuracies) if not a]
        
        ax2.hist(correct_confidences, bins=15, alpha=0.7, label="Correct", color=self.colors[1])
        ax2.hist(incorrect_confidences, bins=15, alpha=0.7, label="Incorrect", color=self.colors[2])
        ax2.set_xlabel("Confidence Score")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Confidence by Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_feedback_analysis(
        self,
        feedback_data: Dict[str, Any],
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot feedback analysis.
        
        Args:
            feedback_data: Dictionary with feedback analysis data.
            figsize: Figure size.
            save_path: Path to save figure (optional).
            
        Returns:
            Matplotlib figure.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Feedback type distribution
        if "feedback_distribution" in feedback_data:
            feedback_types = list(feedback_data["feedback_distribution"].keys())
            feedback_counts = list(feedback_data["feedback_distribution"].values())
            
            ax1.pie(feedback_counts, labels=feedback_types, autopct="%1.1f%%", colors=self.colors)
            ax1.set_title("Feedback Type Distribution")
        
        # Accuracy by feedback type
        if "feedback_type_accuracies" in feedback_data:
            types = list(feedback_data["feedback_type_accuracies"].keys())
            accuracies = list(feedback_data["feedback_type_accuracies"].values())
            
            bars = ax2.bar(types, accuracies, color=self.colors[:len(types)])
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Accuracy by Feedback Type")
            ax2.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{acc:.3f}", ha="center", va="bottom")
        
        # Intervention rate over time
        if "intervention_rate" in feedback_data:
            ax3.bar(["Intervention Rate"], [feedback_data["intervention_rate"]], 
                   color=self.colors[3])
            ax3.set_ylabel("Rate")
            ax3.set_title("Human Intervention Rate")
            ax3.set_ylim(0, 1)
        
        # Response time distribution
        if "response_time_metrics" in feedback_data:
            rt_metrics = feedback_data["response_time_metrics"]
            metrics_names = ["Mean", "Median", "Std"]
            metrics_values = [
                rt_metrics.get("mean_response_time", 0),
                rt_metrics.get("median_response_time", 0),
                rt_metrics.get("response_time_std", 0),
            ]
            
            ax4.bar(metrics_names, metrics_values, color=self.colors[4:7])
            ax4.set_ylabel("Time (seconds)")
            ax4.set_title("Response Time Metrics")
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig


class InteractiveVisualizer:
    """Interactive visualizer using Plotly."""
    
    def __init__(self):
        """Initialize interactive visualizer."""
        self.colors = px.colors.qualitative.Set3
    
    def create_explanation_dashboard(
        self,
        explanations: Dict[str, Dict[str, Any]],
        text: str,
        prediction: str,
        confidence: float,
    ) -> go.Figure:
        """Create interactive explanation dashboard.
        
        Args:
            explanations: Dictionary of explanations by method.
            text: Input text.
            prediction: Model prediction.
            confidence: Model confidence.
            
        Returns:
            Plotly figure.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Feature Importance", "Attention Heatmap", 
                          "Method Comparison", "Confidence Analysis"),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Feature importance plot
        if "SHAP" in explanations and "feature_importance" in explanations["SHAP"]:
            features = explanations["SHAP"]["feature_importance"][:10]
            feature_names = [f["feature"] for f in features]
            importances = [f["importance"] for f in features]
            
            fig.add_trace(
                go.Bar(
                    x=importances,
                    y=feature_names,
                    orientation="h",
                    name="SHAP",
                    marker_color=self.colors[0]
                ),
                row=1, col=1
            )
        
        # Attention heatmap
        if "Attention" in explanations and "attention_map" in explanations["Attention"]:
            attention_map = explanations["Attention"]["attention_map"]
            tokens = [item["token"] for item in attention_map]
            attentions = [item["attention"] for item in attention_map]
            
            fig.add_trace(
                go.Heatmap(
                    z=[attentions],
                    x=tokens,
                    y=["Attention"],
                    colorscale="Blues",
                    name="Attention"
                ),
                row=1, col=2
            )
        
        # Method comparison
        methods = list(explanations.keys())
        method_confidences = [explanations[m].get("confidence", 0) for m in methods]
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=method_confidences,
                name="Confidence",
                marker_color=self.colors[1]
            ),
            row=2, col=1
        )
        
        # Confidence analysis
        fig.add_trace(
            go.Scatter(
                x=[confidence],
                y=[1],
                mode="markers",
                marker=dict(size=20, color=self.colors[2]),
                name="Model Confidence",
                text=[f"Confidence: {confidence:.3f}"],
                hovertemplate="%{text}<extra></extra>"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Explanation Dashboard - Prediction: {prediction}",
            showlegend=False,
            height=800,
        )
        
        return fig
    
    def create_collaboration_dashboard(
        self,
        metrics: Dict[str, Any],
        feedback_history: List[Dict[str, Any]],
    ) -> go.Figure:
        """Create interactive collaboration dashboard.
        
        Args:
            metrics: Collaboration metrics.
            feedback_history: History of feedback interactions.
            
        Returns:
            Plotly figure.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Accuracy Over Time", "Confidence Distribution",
                          "Feedback Types", "Intervention Rate"),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "pie"}, {"type": "indicator"}]]
        )
        
        # Accuracy over time
        if feedback_history:
            timestamps = list(range(len(feedback_history)))
            accuracies = [entry.get("is_correct", False) for entry in feedback_history]
            cumulative_accuracy = np.cumsum(accuracies) / np.arange(1, len(accuracies) + 1)
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cumulative_accuracy,
                    mode="lines+markers",
                    name="Cumulative Accuracy",
                    line=dict(color=self.colors[0])
                ),
                row=1, col=1
            )
        
        # Confidence distribution
        if feedback_history:
            confidences = [entry.get("confidence", 0) for entry in feedback_history]
            
            fig.add_trace(
                go.Histogram(
                    x=confidences,
                    name="Confidence Distribution",
                    marker_color=self.colors[1]
                ),
                row=1, col=2
            )
        
        # Feedback types
        if "feedback_distribution" in metrics:
            feedback_types = list(metrics["feedback_distribution"].keys())
            feedback_counts = list(metrics["feedback_distribution"].values())
            
            fig.add_trace(
                go.Pie(
                    labels=feedback_types,
                    values=feedback_counts,
                    name="Feedback Types",
                    marker_colors=self.colors[2:2+len(feedback_types)]
                ),
                row=2, col=1
            )
        
        # Intervention rate
        intervention_rate = metrics.get("intervention_rate", 0)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=intervention_rate,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Intervention Rate"},
                gauge={
                    "axis": {"range": [None, 1]},
                    "bar": {"color": self.colors[3]},
                    "steps": [
                        {"range": [0, 0.3], "color": "lightgray"},
                        {"range": [0.3, 0.7], "color": "gray"},
                        {"range": [0.7, 1], "color": "darkgray"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 0.8
                    }
                }
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Human-AI Collaboration Dashboard",
            showlegend=False,
            height=800,
        )
        
        return fig
