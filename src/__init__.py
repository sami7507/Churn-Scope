"""
Churn Prediction System — Source Package
Provides shared utilities: config loading and structured logging.
"""

import logging
import logging.handlers
import os
from pathlib import Path

import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_logger(name: str, config: dict = None) -> logging.Logger:
    """Create a structured logger with console and rotating file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_level = logging.INFO
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    log_file = "logs/churn_prediction.log"

    if config and "logging" in config:
        log_cfg = config["logging"]
        log_level = getattr(logging, log_cfg.get("level", "INFO").upper())
        log_format = log_cfg.get("format", log_format)
        log_file = log_cfg.get("file", log_file)

    logger.setLevel(log_level)
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10_485_760, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
