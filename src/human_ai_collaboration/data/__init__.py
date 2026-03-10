"""Data loading and preprocessing utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from ..utils import set_seed

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for text classification tasks."""
    
    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        random_seed: int = 42,
    ):
        """Initialize data loader.
        
        Args:
            data_path: Path to data directory.
            tokenizer_name: Name of the tokenizer to use.
            max_length: Maximum sequence length.
            random_seed: Random seed for reproducibility.
        """
        self.data_path = Path(data_path) if data_path else None
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.random_seed = random_seed
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label_encoder = LabelEncoder()
        
        set_seed(random_seed)
    
    def load_20newsgroups(
        self,
        subset: str = "all",
        categories: Optional[List[str]] = None,
        remove_headers: bool = True,
        remove_footers: bool = True,
        remove_quotes: bool = True,
    ) -> Tuple[List[str], np.ndarray, List[str]]:
        """Load 20 Newsgroups dataset.
        
        Args:
            subset: Dataset subset ('train', 'test', 'all').
            categories: List of categories to include.
            remove_headers: Whether to remove headers.
            remove_footers: Whether to remove footers.
            remove_quotes: Whether to remove quotes.
            
        Returns:
            Tuple of (texts, labels, label_names).
        """
        logger.info("Loading 20 Newsgroups dataset...")
        
        remove = []
        if remove_headers:
            remove.append("headers")
        if remove_footers:
            remove.append("footers")
        if remove_quotes:
            remove.append("quotes")
        
        newsgroups = fetch_20newsgroups(
            subset=subset,
            categories=categories,
            remove=tuple(remove) if remove else None,
        )
        
        texts = newsgroups.data
        labels = newsgroups.target
        label_names = newsgroups.target_names
        
        logger.info(f"Loaded {len(texts)} texts with {len(label_names)} categories")
        
        return texts, labels, label_names
    
    def load_custom_data(
        self,
        texts: List[str],
        labels: List[Union[str, int]],
        label_names: Optional[List[str]] = None,
    ) -> Tuple[List[str], np.ndarray, List[str]]:
        """Load custom text data.
        
        Args:
            texts: List of text documents.
            labels: List of labels (strings or integers).
            label_names: Optional list of label names.
            
        Returns:
            Tuple of (texts, encoded_labels, label_names).
        """
        logger.info(f"Loading custom dataset with {len(texts)} texts...")
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        if label_names is None:
            label_names = self.label_encoder.classes_.tolist()
        
        logger.info(f"Loaded {len(texts)} texts with {len(label_names)} categories")
        
        return texts, encoded_labels, label_names
    
    def tokenize_texts(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize texts for model input.
        
        Args:
            texts: List of texts to tokenize.
            padding: Whether to pad sequences.
            truncation: Whether to truncate sequences.
            
        Returns:
            Dictionary with tokenized inputs.
        """
        logger.info(f"Tokenizing {len(texts)} texts...")
        
        tokenized = self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return tokenized
    
    def create_splits(
        self,
        texts: List[str],
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True,
    ) -> Tuple[
        Tuple[List[str], np.ndarray],
        Tuple[List[str], np.ndarray],
        Tuple[List[str], np.ndarray],
    ]:
        """Create train/validation/test splits.
        
        Args:
            texts: List of texts.
            labels: Array of labels.
            test_size: Proportion of data for test set.
            val_size: Proportion of data for validation set.
            stratify: Whether to stratify splits.
            
        Returns:
            Tuple of (train, val, test) splits.
        """
        logger.info("Creating train/validation/test splits...")
        
        # First split: train+val vs test
        stratify_labels = labels if stratify else None
        texts_temp, texts_test, labels_temp, labels_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=self.random_seed,
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_labels_temp = labels_temp if stratify else None
        texts_train, texts_val, labels_train, labels_val = train_test_split(
            texts_temp,
            labels_temp,
            test_size=val_size_adjusted,
            stratify=stratify_labels_temp,
            random_state=self.random_seed,
        )
        
        logger.info(
            f"Created splits - Train: {len(texts_train)}, "
            f"Val: {len(texts_val)}, Test: {len(texts_test)}"
        )
        
        return (
            (texts_train, labels_train),
            (texts_val, labels_val),
            (texts_test, labels_test),
        )
    
    def save_metadata(
        self,
        label_names: List[str],
        output_path: Union[str, Path],
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save dataset metadata.
        
        Args:
            label_names: List of label names.
            output_path: Path to save metadata.
            additional_info: Additional metadata to save.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "label_names": label_names,
            "num_classes": len(label_names),
            "tokenizer_name": self.tokenizer_name,
            "max_length": self.max_length,
            "random_seed": self.random_seed,
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {output_path}")


class SyntheticDataGenerator:
    """Generate synthetic text data for testing and demonstration."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize synthetic data generator.
        
        Args:
            random_seed: Random seed for reproducibility.
        """
        self.random_seed = random_seed
        set_seed(random_seed)
    
    def generate_text_classification_data(
        self,
        n_samples: int = 1000,
        n_classes: int = 5,
        text_length_range: Tuple[int, int] = (50, 200),
        topics: Optional[List[str]] = None,
    ) -> Tuple[List[str], np.ndarray, List[str]]:
        """Generate synthetic text classification data.
        
        Args:
            n_samples: Number of samples to generate.
            n_classes: Number of classes.
            text_length_range: Range of text lengths.
            topics: Optional list of topics for each class.
            
        Returns:
            Tuple of (texts, labels, label_names).
        """
        logger.info(f"Generating synthetic data: {n_samples} samples, {n_classes} classes")
        
        if topics is None:
            topics = [
                "technology", "science", "health", "business", "entertainment",
                "sports", "politics", "education", "travel", "food"
            ][:n_classes]
        
        texts = []
        labels = []
        
        # Generate texts for each class
        samples_per_class = n_samples // n_classes
        
        for class_idx in range(n_classes):
            topic = topics[class_idx]
            
            for _ in range(samples_per_class):
                text_length = np.random.randint(*text_length_range)
                text = self._generate_text_for_topic(topic, text_length)
                texts.append(text)
                labels.append(class_idx)
        
        # Add remaining samples to make exact count
        remaining = n_samples - len(texts)
        for _ in range(remaining):
            class_idx = np.random.randint(n_classes)
            topic = topics[class_idx]
            text_length = np.random.randint(*text_length_range)
            text = self._generate_text_for_topic(topic, text_length)
            texts.append(text)
            labels.append(class_idx)
        
        labels = np.array(labels)
        
        logger.info(f"Generated {len(texts)} texts with {n_classes} classes")
        
        return texts, labels, topics
    
    def _generate_text_for_topic(self, topic: str, length: int) -> str:
        """Generate text for a specific topic.
        
        Args:
            topic: Topic to generate text about.
            length: Desired text length.
            
        Returns:
            Generated text.
        """
        # Simple template-based text generation
        templates = {
            "technology": [
                f"The latest developments in {topic} are revolutionizing the industry.",
                f"Many companies are investing heavily in {topic} research and development.",
                f"The future of {topic} looks promising with new innovations emerging.",
                f"Experts predict significant growth in the {topic} sector.",
            ],
            "science": [
                f"Recent studies in {topic} have revealed fascinating insights.",
                f"Scientists are making breakthrough discoveries in {topic} research.",
                f"The field of {topic} continues to advance our understanding.",
                f"New methodologies in {topic} are opening up exciting possibilities.",
            ],
            "health": [
                f"Healthcare professionals are focusing on {topic} treatments.",
                f"New research shows promising results for {topic} interventions.",
                f"The importance of {topic} in maintaining good health cannot be overstated.",
                f"Medical advances in {topic} are improving patient outcomes.",
            ],
            "business": [
                f"The {topic} market is experiencing significant growth this quarter.",
                f"Business leaders are optimistic about {topic} opportunities.",
                f"Investment in {topic} is driving economic development.",
                f"The {topic} industry is adapting to changing market conditions.",
            ],
            "entertainment": [
                f"The entertainment industry is embracing {topic} in new ways.",
                f"Audiences are responding positively to {topic} content.",
                f"Creative professionals are exploring {topic} themes.",
                f"The {topic} genre continues to evolve and attract fans.",
            ],
        }
        
        # Choose template category based on topic
        category = "technology"  # Default
        for cat in templates:
            if cat in topic.lower():
                category = cat
                break
        
        template = np.random.choice(templates[category])
        
        # Add some random words to reach desired length
        words = template.split()
        while len(" ".join(words)) < length:
            filler_words = [
                "important", "significant", "notable", "remarkable", "outstanding",
                "innovative", "cutting-edge", "advanced", "sophisticated", "modern"
            ]
            words.insert(np.random.randint(len(words)), np.random.choice(filler_words))
        
        return " ".join(words)
