"""
Utility functions.
"""

import numpy as np
import random
import torch
from typing import Tuple


def get_bbox_from_mask(mask: np.ndarray, padding: int = 10) -> Tuple[int, int, int, int]:
    """
    Extract bounding box from binary mask.
    
    Args:
        mask: Binary mask (H, W)
        padding: Padding around the bbox
    
    Returns:
        Bounding box (x_min, y_min, x_max, y_max)
    """
    if mask.sum() == 0:
        H, W = mask.shape
        return (0, 0, W, H)
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    H, W = mask.shape
    
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(W, x_max + padding + 1)
    y_max = min(H, y_max + padding + 1)
    
    return (x_min, y_min, x_max, y_max)


def setup_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)