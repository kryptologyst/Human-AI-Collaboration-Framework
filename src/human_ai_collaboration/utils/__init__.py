"""Utility functions for the Human-AI Collaboration Framework."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level.
        log_file: Optional log file path.
        format_string: Custom format string.
        
    Returns:
        Configured logger.
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else []),
        ],
    )
    
    return logging.getLogger(__name__)


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        output_path: Output file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(OmegaConf.to_yaml(config), f, default_flow_style=False)


def validate_config(config: DictConfig, required_keys: list[str]) -> None:
    """Validate configuration has required keys.
    
    Args:
        config: Configuration to validate.
        required_keys: List of required configuration keys.
        
    Raises:
        ValueError: If required keys are missing.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """Get model size information.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Dictionary with model size information.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
    }
