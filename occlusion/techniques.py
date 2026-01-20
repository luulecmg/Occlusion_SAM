# occlusion/techniques.py
"""
Occlusion techniques for robustness evaluation.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path
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


class SurgicalTool(BaseOcclusion):
    """Occlusion using extracted surgical tools from dataset."""
    
    def __init__(self, tools_dir: Optional[str] = None, ratio: float = 0.2):
        """
        Args:
            tools_dir: Directory containing extracted tool images (.png files)
            ratio: Occlusion ratio (determines tool size scaling)
        """
        super().__init__(ratio)
        self.name = "surgical_tool"
        self.tools_dir = Path(tools_dir) if tools_dir else Path("./outputs/extracted_tools")
        self.tools = self._load_tools()
        
        if not self.tools:
            raise ValueError(f"No tool images found in {self.tools_dir}")
    
    def _load_tools(self) -> list:
        """Load all tool images from directory."""
        tools = []
        if self.tools_dir.exists():
            for tool_path in self.tools_dir.glob("*.png"):
                try:
                    tool = cv2.imread(str(tool_path), cv2.IMREAD_UNCHANGED)
                    if tool is not None:
                        tools.append(tool)
                except Exception as e:
                    print(f"Warning: Could not load tool from {tool_path}: {e}")
        return tools
    
    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Dict:
        """Apply surgical tool occlusion.
        
        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W)
            bbox: Bounding box (x_min, y_min, x_max, y_max)
        
        Returns:
            Dict with 'image', 'mask', 'occlusion_mask'
        """
        image_out = image.copy()
        mask_out = mask.copy()
        
        if not self.tools:
            return {
                'image': image_out,
                'mask': mask_out,
                'occlusion_mask': np.zeros_like(mask)
            }
        
        H, W = image.shape[:2]
        x_min, y_min, x_max, y_max = bbox
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        # Calculate tool size based on ratio
        tool_w = int(bbox_w * np.sqrt(self.ratio))
        tool_h = int(bbox_h * np.sqrt(self.ratio))
        
        if tool_w < 1 or tool_h < 1:
            return {
                'image': image_out,
                'mask': mask_out,
                'occlusion_mask': np.zeros_like(mask)
            }
        
        # Random position inside bbox
        cx = np.random.randint(x_min, max(x_min + 1, x_max - tool_w))
        cy = np.random.randint(y_min, max(y_min + 1, y_max - tool_h))
        
        # Select random tool
        tool = self.tools[np.random.randint(0, len(self.tools))]
        
        # Resize tool to calculated size
        tool_resized = cv2.resize(tool, (tool_w, tool_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply tool (with alpha blending if it has alpha channel)
        occlusion_mask = np.zeros_like(mask)
        
        try:
            if tool_resized.shape[2] == 4:  # RGBA
                alpha = tool_resized[:, :, 3].astype(float) / 255.0
                tool_rgb = tool_resized[:, :, :3]
                
                # Alpha blend
                for c in range(3):
                    image_out[cy:cy+tool_h, cx:cx+tool_w, c] = (
                        (1 - alpha) * image_out[cy:cy+tool_h, cx:cx+tool_w, c] +
                        alpha * tool_rgb[:, :, c]
                    )
                
                # Occlusion mask is where alpha > 0.5
                occlusion_mask[cy:cy+tool_h, cx:cx+tool_w] = (alpha > 0.5).astype(np.uint8)
            else:  # RGB or grayscale
                # Direct replacement
                tool_rgb = tool_resized[:, :, :3] if len(tool_resized.shape) > 2 else cv2.cvtColor(tool_resized, cv2.COLOR_GRAY2RGB)
                image_out[cy:cy+tool_h, cx:cx+tool_w] = tool_rgb
                occlusion_mask[cy:cy+tool_h, cx:cx+tool_w] = 1
        except Exception as e:
            print(f"Warning: Could not apply tool: {e}")
        
        return {
            'image': image_out,
            'mask': mask_out,  # Ground truth mask unchanged
            'occlusion_mask': occlusion_mask
        }


def get_occlusion(name: str, ratio: float = 0.0, tools_dir: Optional[str] = None) -> BaseOcclusion:
    """
    Factory function to get occlusion technique.
    
    Args:
        name: 'none', 'cutout', 'cutmix', or 'surgical_tool'
        ratio: Occlusion ratio
        tools_dir: Directory for surgical tool (required if name='surgical_tool')
    
    Returns:
        Occlusion instance
    """
    techniques = {
        'none': NoOcclusion,
        'cutout': Cutout,
        'cutmix': Cutmix,
        'surgical_tool': SurgicalTool,
    }
    
    if name.lower() not in techniques:
        raise ValueError(f"Unknown occlusion: {name}. Available: {list(techniques.keys())}")
    
    if name.lower() == 'none':
        return NoOcclusion()
    elif name.lower() == 'surgical_tool':
        return SurgicalTool(tools_dir=tools_dir, ratio=ratio)
    
    return techniques[name.lower()](ratio=ratio)