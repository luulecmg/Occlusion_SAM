# occlusion/techniques.py
"""
Occlusion techniques for robustness evaluation.
"""

import numpy as np
from typing import Tuple, Dict
import cv2


class BaseOcclusion:
    """Base class for occlusion techniques."""
    
    def __init__(self, ratio: float = 0.0):
        """
        Args:
            ratio: Occlusion ratio (0.0 to 1.0)
        """
        self.ratio = ratio
        self.name = "base"
    
    def __call__(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Dict:
        """
        Apply occlusion.
        
        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W)
            bbox: Bounding box (x_min, y_min, x_max, y_max)
        
        Returns:
            Dict with 'image', 'mask', 'occlusion_mask'
        """
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.name}(ratio={self.ratio})"


class NoOcclusion(BaseOcclusion):
    """No occlusion (baseline)."""
    
    def __init__(self):
        super().__init__(ratio=0.0)
        self.name = "none"
    
    def __call__(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Dict:
        return {
            'image': image.copy(),
            'mask': mask.copy(),
            'occlusion_mask': np.zeros_like(mask)
        }


class Cutout(BaseOcclusion):
    """
    Cutout occlusion: Remove a random square region inside bbox.
    """
    
    def __init__(self, ratio: float = 0.2):
        super().__init__(ratio)
        self.name = "cutout"
    
    def __call__(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Dict:
        image_out = image.copy()
        mask_out = mask.copy()
        
        x_min, y_min, x_max, y_max = bbox
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        # Calculate cutout size based on ratio
        cut_w = int(bbox_w * np.sqrt(self.ratio))
        cut_h = int(bbox_h * np.sqrt(self.ratio))
        
        if cut_w < 1 or cut_h < 1:
            return {
                'image': image_out,
                'mask': mask_out,
                'occlusion_mask': np.zeros_like(mask)
            }
        
        # Random position inside bbox
        cx = np.random.randint(x_min, max(x_min + 1, x_max - cut_w))
        cy = np.random.randint(y_min, max(y_min + 1, y_max - cut_h))
        
        # Apply cutout (fill with black/zero)
        image_out[cy:cy+cut_h, cx:cx+cut_w] = 0
        
        # Create occlusion mask
        occlusion_mask = np.zeros_like(mask)
        occlusion_mask[cy:cy+cut_h, cx:cx+cut_w] = 1
        
        return {
            'image': image_out,
            'mask': mask_out,  # Ground truth mask unchanged
            'occlusion_mask': occlusion_mask
        }


class Cutmix(BaseOcclusion):
    """
    Cutmix occlusion: Replace a region with content from outside bbox.
    """
    
    def __init__(self, ratio: float = 0.2):
        super().__init__(ratio)
        self.name = "cutmix"
    
    def __call__(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Dict:
        image_out = image.copy()
        mask_out = mask.copy()
        
        H, W = image.shape[:2]
        x_min, y_min, x_max, y_max = bbox
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        # Calculate cutmix size
        cut_w = int(bbox_w * np.sqrt(self.ratio))
        cut_h = int(bbox_h * np.sqrt(self.ratio))
        
        if cut_w < 1 or cut_h < 1:
            return {
                'image': image_out,
                'mask': mask_out,
                'occlusion_mask': np.zeros_like(mask)
            }
        
        # Random position inside bbox (where to paste)
        cx = np.random.randint(x_min, max(x_min + 1, x_max - cut_w))
        cy = np.random.randint(y_min, max(y_min + 1, y_max - cut_h))
        
        # Get source patch from outside bbox (background region)
        # Try to find a region outside the mask
        source_patch = self._get_background_patch(image, mask, cut_h, cut_w)
        
        # Apply cutmix
        image_out[cy:cy+cut_h, cx:cx+cut_w] = source_patch
        
        # Create occlusion mask
        occlusion_mask = np.zeros_like(mask)
        occlusion_mask[cy:cy+cut_h, cx:cx+cut_w] = 1
        
        return {
            'image': image_out,
            'mask': mask_out,
            'occlusion_mask': occlusion_mask
        }
    
    def _get_background_patch(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        h: int, 
        w: int
    ) -> np.ndarray:
        """Get a patch from background region."""
        H, W = image.shape[:2]
        
        # Try to find background region
        for _ in range(10):
            y = np.random.randint(0, max(1, H - h))
            x = np.random.randint(0, max(1, W - w))
            
            # Check if mostly background
            patch_mask = mask[y:y+h, x:x+w]
            if patch_mask.sum() < 0.3 * h * w:
                return image[y:y+h, x:x+w].copy()
        
        # Fallback: return random patch
        y = np.random.randint(0, max(1, H - h))
        x = np.random.randint(0, max(1, W - w))
        return image[y:y+h, x:x+w].copy()


def get_occlusion(name: str, ratio: float = 0.0) -> BaseOcclusion:
    """
    Factory function to get occlusion technique.
    
    Args:
        name: 'none', 'cutout', or 'cutmix'
        ratio: Occlusion ratio
    
    Returns:
        Occlusion instance
    """
    techniques = {
        'none': NoOcclusion,
        'cutout': Cutout,
        'cutmix': Cutmix,
    }
    
    if name.lower() not in techniques:
        raise ValueError(f"Unknown occlusion: {name}. Available: {list(techniques.keys())}")
    
    if name.lower() == 'none':
        return NoOcclusion()
    
    return techniques[name.lower()](ratio=ratio)