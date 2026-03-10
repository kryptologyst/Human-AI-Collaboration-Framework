"""Human-AI Collaboration Framework for Explainable AI.

This package provides a comprehensive framework for human-AI collaboration
in text classification tasks, with emphasis on transparency, interpretability,
and continuous learning from human feedback.
"""

__version__ = "1.0.0"
__author__ = "XAI Research Team"

from .models import TextClassifier, CollaborationModel
from .explainers import (
    SHAPExplainer,
    LIMEExplainer,
    AttentionExplainer,
    IntegratedGradientsExplainer,
)
from .metrics import (
    CollaborationMetrics,
    ExplanationMetrics,
    TrustMetrics,
    FairnessMetrics,
)
from .data import DataLoader, SyntheticDataGenerator
from .utils import (
    set_seed,
    get_device,
    load_config,
    setup_logging,
)

__all__ = [
    "TextClassifier",
    "CollaborationModel",
    "SHAPExplainer",
    "LIMEExplainer",
    "AttentionExplainer",
    "IntegratedGradientsExplainer",
    "CollaborationMetrics",
    "ExplanationMetrics",
    "TrustMetrics",
    "FairnessMetrics",
    "DataLoader",
    "SyntheticDataGenerator",
    "set_seed",
    "get_device",
    "load_config",
    "setup_logging",
]
