#!/usr/bin/env python3
"""Evaluation script for Human-AI Collaboration Framework."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
import yaml

from human_ai_collaboration import (
    TextClassifier,
    CollaborationModel,
    DataLoader,
    SHAPExplainer,
    LIMEExplainer,
    CollaborationMetrics,
    ExplanationMetrics,
    set_seed,
    load_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Human-AI Collaboration Model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/",
        help="Path to data directory"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Output directory for evaluation results"
    )
    
    parser.add_argument(
        "--explanation_methods",
        type=str,
        nargs="+",
        default=["SHAP", "LIME"],
        help="Explanation methods to evaluate"
    )
    
    return parser.parse_args()


def load_model(model_path: str, config: Dict[str, Any]) -> TextClassifier:
    """Load trained model.
    
    Args:
        model_path: Path to model checkpoint.
        config: Model configuration.
        
    Returns:
        Loaded text classifier.
    """
    model = TextClassifier(
        model_name=config["name"],
        num_classes=config["num_classes"],
        dropout_rate=config.get("dropout_rate", 0.1),
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model


def evaluate_collaboration(
    model: TextClassifier,
    test_data: tuple,
    label_names: list,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate collaboration performance.
    
    Args:
        model: Trained text classifier.
        test_data: Test data tuple.
        label_names: List of label names.
        config: Collaboration configuration.
        
    Returns:
        Dictionary with collaboration metrics.
    """
    test_texts, test_labels = test_data
    
    # Create collaboration model
    collab_model = CollaborationModel(
        base_model=model,
        confidence_threshold=config["confidence_threshold"],
    )
    
    # Initialize metrics
    metrics = CollaborationMetrics()
    
    # Simulate collaboration
    for i, text in enumerate(test_texts[:100]):  # Limit for demo
        # Get AI suggestion
        suggested_label, confidence = collab_model.suggest_label(text, label_names)
        
        # Simulate human feedback (accept if correct, reject if wrong)
        true_label = label_names[test_labels[i]]
        is_correct = suggested_label == true_label
        feedback_type = "accept" if is_correct else "reject"
        
        # Process feedback
        collab_model.process_feedback(
            text=text,
            ai_suggestion=suggested_label,
            human_label=true_label,
            confidence=confidence,
            feedback_type=feedback_type,
        )
        
        # Update metrics
        metrics.update(
            prediction=suggested_label,
            ground_truth=true_label,
            confidence=confidence,
            feedback_type=feedback_type,
        )
    
    return metrics.get_summary()


def evaluate_explanations(
    model: TextClassifier,
    test_data: tuple,
    explanation_methods: list,
) -> Dict[str, Any]:
    """Evaluate explanation quality.
    
    Args:
        model: Trained text classifier.
        test_data: Test data tuple.
        explanation_methods: List of explanation methods to evaluate.
        
    Returns:
        Dictionary with explanation metrics.
    """
    test_texts, test_labels = test_data
    
    # Initialize explainers
    explainers = {}
    if "SHAP" in explanation_methods:
        explainers["SHAP"] = SHAPExplainer(model)
    if "LIME" in explanation_methods:
        explainers["LIME"] = LIMEExplainer(model)
    
    # Initialize metrics
    metrics = ExplanationMetrics()
    
    # Evaluate explanations
    for i, text in enumerate(test_texts[:50]):  # Limit for demo
        for method, explainer in explainers.items():
            try:
                explanation = explainer.explain(text)
                metrics.update(explanation, method=method)
            except Exception as e:
                logger.warning(f"Failed to generate {method} explanation: {e}")
    
    return metrics.get_summary()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.log_file,
    )
    
    # Set random seed
    set_seed(config.data.random_seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Human-AI Collaboration Model Evaluation")
    
    try:
        # Load model
        logger.info("Loading model...")
        model = load_model(args.model_path, config.model)
        
        # Prepare test data
        logger.info("Preparing test data...")
        loader = DataLoader(
            data_path=args.data_path,
            tokenizer_name=config.data.tokenizer_name,
            max_length=config.data.max_length,
            random_seed=config.data.random_seed,
        )
        
        if config.data.dataset_name == "20newsgroups":
            texts, labels, label_names = loader.load_20newsgroups(subset="test")
        else:
            from human_ai_collaboration import SyntheticDataGenerator
            generator = SyntheticDataGenerator(random_seed=config.data.random_seed)
            texts, labels, label_names = generator.generate_text_classification_data(
                n_samples=200, n_classes=config.model.num_classes
            )
        
        test_data = (texts, labels)
        
        # Evaluate collaboration
        logger.info("Evaluating collaboration performance...")
        collaboration_metrics = evaluate_collaboration(
            model, test_data, label_names, config.collaboration
        )
        
        # Evaluate explanations
        logger.info("Evaluating explanation quality...")
        explanation_metrics = evaluate_explanations(
            model, test_data, args.explanation_methods
        )
        
        # Combine results
        results = {
            "collaboration_metrics": collaboration_metrics,
            "explanation_metrics": explanation_metrics,
            "model_info": {
                "model_path": args.model_path,
                "num_classes": config.model.num_classes,
                "explanation_methods": args.explanation_methods,
            },
        }
        
        # Save results
        with open(output_dir / "evaluation_results.yaml", "w") as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to {output_dir}")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if "accuracy_metrics" in collaboration_metrics:
            acc_metrics = collaboration_metrics["accuracy_metrics"]
            print(f"Overall Accuracy: {acc_metrics.get('overall_accuracy', 0):.4f}")
            print(f"High Confidence Accuracy: {acc_metrics.get('high_confidence_accuracy', 0):.4f}")
        
        if "confidence_metrics" in collaboration_metrics:
            conf_metrics = collaboration_metrics["confidence_metrics"]
            print(f"Mean Confidence: {conf_metrics.get('mean_confidence', 0):.4f}")
            print(f"Calibration Error: {conf_metrics.get('calibration_error', 0):.4f}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
