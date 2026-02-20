"""Configuration utilities for loading and managing YAML configs."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Successfully loaded config from {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save.
        save_path: Path where to save the YAML file.
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    with open(save_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved to {save_path}")


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base_config: Base configuration dictionary.
        override_config: Configuration to override base values.

    Returns:
        Merged configuration dictionary.
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def get_nested_value(
    config: Dict[str, Any],
    key_path: str,
    default: Optional[Any] = None
) -> Any:
    """
    Get a value from nested dictionary using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to the value (e.g., 'model.hidden_dim').
        default: Default value if key doesn't exist.

    Returns:
        Value at the specified path, or default if not found.
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
