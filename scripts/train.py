#!/usr/bin/env python3
"""Training script for Human-AI Collaboration Framework."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
import yaml

from human_ai_collaboration import (
    TextClassifier,
    DataLoader as HACDataLoader,
    set_seed,
    get_device,
    load_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Human-AI Collaboration Model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/",
        help="Path to data directory"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Name of the pre-trained model"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/",
        help="Output directory for model checkpoints"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def create_model(config: Dict[str, Any]) -> TextClassifier:
    """Create and initialize the model.
    
    Args:
        config: Model configuration.
        
    Returns:
        Initialized text classifier.
    """
    model = TextClassifier(
        model_name=config["name"],
        num_classes=config["num_classes"],
        dropout_rate=config.get("dropout_rate", 0.1),
        freeze_encoder=config.get("freeze_encoder", False),
    )
    
    return model


def prepare_data(config: Dict[str, Any], data_path: str) -> tuple:
    """Prepare training data.
    
    Args:
        config: Data configuration.
        data_path: Path to data directory.
        
    Returns:
        Tuple of (train_data, val_data, test_data, label_names).
    """
    loader = HACDataLoader(
        data_path=data_path,
        tokenizer_name=config["tokenizer_name"],
        max_length=config["max_length"],
        random_seed=config["random_seed"],
    )
    
    # Load dataset
    if config["dataset_name"] == "20newsgroups":
        texts, labels, label_names = loader.load_20newsgroups(subset="all")
    else:
        # Use synthetic data for demonstration
        from human_ai_collaboration import SyntheticDataGenerator
        generator = SyntheticDataGenerator(random_seed=config["random_seed"])
        texts, labels, label_names = generator.generate_text_classification_data(
            n_samples=1000, n_classes=config.get("num_classes", 5)
        )
    
    # Create splits
    train_data, val_data, test_data = loader.create_splits(
        texts, labels,
        test_size=config.get("test_size", 0.2),
        val_size=config.get("val_size", 0.1),
    )
    
    # Save metadata
    loader.save_metadata(
        label_names,
        Path(data_path) / "metadata.json",
        additional_info={"num_samples": len(texts)}
    )
    
    return train_data, val_data, test_data, label_names


def train_model(
    model: TextClassifier,
    train_data: tuple,
    val_data: tuple,
    config: Dict[str, Any],
    output_dir: str,
) -> None:
    """Train the model.
    
    Args:
        model: Text classifier model.
        train_data: Training data tuple.
        val_data: Validation data tuple.
        config: Training configuration.
        output_dir: Output directory for checkpoints.
    """
    train_texts, train_labels = train_data
    val_texts, val_labels = val_data
    
    # Tokenize data
    train_encodings = model.tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=config["max_length"],
    )
    
    val_encodings = model.tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=config["max_length"],
    )
    
    # Create datasets
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
    
    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("max_epochs", 10),
        per_device_train_batch_size=config.get("batch_size", 32),
        per_device_eval_batch_size=config.get("batch_size", 32),
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    logger.info(f"Model saved to {output_dir}")


def evaluate_model(
    model: TextClassifier,
    test_data: tuple,
    label_names: list,
) -> Dict[str, float]:
    """Evaluate the trained model.
    
    Args:
        model: Trained text classifier.
        test_data: Test data tuple.
        label_names: List of label names.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    test_texts, test_labels = test_data
    
    # Get predictions
    predictions, probabilities = model.predict(test_texts)
    
    # Calculate accuracy
    accuracy = (predictions == test_labels).mean()
    
    # Calculate confidence metrics
    max_confidences = np.max(probabilities, axis=1)
    avg_confidence = np.mean(max_confidences)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Average Confidence: {avg_confidence:.4f}")
    
    return {
        "accuracy": float(accuracy),
        "avg_confidence": float(avg_confidence),
        "num_test_samples": len(test_texts),
    }


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_name:
        config.model.name = args.model_name
    if args.num_epochs:
        config.model.max_epochs = args.num_epochs
    if args.batch_size:
        config.model.batch_size = args.batch_size
    if args.learning_rate:
        config.model.learning_rate = args.learning_rate
    if args.random_seed:
        config.data.random_seed = args.random_seed
    
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
    
    logger.info("Starting Human-AI Collaboration Model Training")
    logger.info(f"Configuration: {config}")
    
    try:
        # Prepare data
        logger.info("Preparing data...")
        train_data, val_data, test_data, label_names = prepare_data(
            config.data, args.data_path
        )
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config.model)
        
        # Train model
        logger.info("Training model...")
        train_model(model, train_data, val_data, config.model, str(output_dir))
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, test_data, label_names)
        
        # Save evaluation results
        with open(output_dir / "evaluation_results.yaml", "w") as f:
            yaml.dump(metrics, f, default_flow_style=False)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
