"""
Logging Utilities
src/utils/logger.py
"""

import logging
import sys
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def setup_logger(log_file: Path) -> logging.Logger:
    """Set up file and console logger"""
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger


class TensorboardLogger:
    """Wrapper for TensorBoard SummaryWriter"""
    
    def __init__(self, log_dir: Path):
        self.writer = SummaryWriter(log_dir)
        
    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        self.writer.add_histogram(tag, values, step)
        
    def close(self):
        self.writer.close()
