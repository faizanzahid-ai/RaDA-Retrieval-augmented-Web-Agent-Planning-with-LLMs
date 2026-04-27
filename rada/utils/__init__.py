"""
Utility functions
"""

import os
import json
from pathlib import Path
from loguru import logger

from .prompts import (
    TASK_DECOMPOSITION_PROMPT,
    ACTION_GENERATION_PROMPT,
    EXEMPLAR_RETRIEVAL_PROMPT,
    SUCCESS_VERIFICATION_PROMPT,
)


def setup_logging(level: str = "INFO"):
    """Setup logging"""
    logger.add(
        "logs/rada.log",
        rotation="10 MB",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.info("Logging initialized")


def load_config(config_path: str = ".env") -> dict:
    """Load configuration"""
    config = {}
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    config[key] = value
    return config


def ensure_dirs():
    """Ensure required directories exist"""
    dirs = ['data', 'logs', 'screenshots', 'evaluation_results']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def format_metrics(metrics: dict) -> str:
    """Format metrics for display"""
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if value < 1.0:
                lines.append(f"{key}: {value:.2%}")
            else:
                lines.append(f"{key}: {value:.2f}")
        else:
            lines.append(f"{key}: {value}")
    return '\n'.join(lines)


def save_results(results: dict, path: str):
    """Save results to JSON"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {path}")


__all__ = [
    "TASK_DECOMPOSITION_PROMPT",
    "ACTION_GENERATION_PROMPT",
    "EXEMPLAR_RETRIEVAL_PROMPT",
    "SUCCESS_VERIFICATION_PROMPT",
    "setup_logging",
    "load_config",
    "ensure_dirs",
    "format_metrics",
    "save_results",
]
