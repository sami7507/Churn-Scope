"""ChurnScope — Source Package. Shared utilities: config loading and logging."""
import logging, logging.handlers, os
from pathlib import Path
import yaml

def load_config(config_path: str = "config/config.yaml") -> dict:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(p) as f:
        return yaml.safe_load(f)

def get_logger(name: str, config: dict = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    log_level = logging.INFO
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    log_file = "logs/churn_prediction.log"
    if config and "logging" in config:
        lc = config["logging"]
        log_level  = getattr(logging, lc.get("level","INFO").upper())
        log_format = lc.get("format", log_format)
        log_file   = lc.get("file",   log_file)
    logger.setLevel(log_level)
    fmt = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=10_485_760, backupCount=5)
    fh.setFormatter(fmt); logger.addHandler(fh)
    return logger
