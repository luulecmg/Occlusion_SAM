"""
Utility functions.
"""

import numpy as np
import random
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional


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


def extract_tools_from_masks(
    dataset_path: str,
    num_samples: int = 5,
    save_dir: Optional[str] = None
) -> None:
    """Extract surgical tools using mask segmentation with cropping.
    
    Args:
        dataset_path: Path to dataset with 'images' and 'masks' subdirectories
        num_samples: Number of samples to process and visualize
        save_dir: Directory to save extracted tools (optional)
    """
    dataset_path = Path(dataset_path).resolve()
    images_dir = dataset_path / "images" / "images"
    masks_dir = dataset_path / "masks" / "masks"
    
    if save_dir:
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Save dir: {save_dir}")
    
    # Get list of image files
    image_files = sorted(list(images_dir.glob("*.jpg")))[:num_samples]
    print(f"Found {len(image_files)} images")
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(image_files):
        img_name = img_path.stem
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Load corresponding mask
        mask_path = masks_dir / f"{img_name}.png"
        if not mask_path.exists():
            # Try other extensions
            for ext in ['.png', '.jpg', '.JPG', '.PNG']:
                mask_path = masks_dir / f"{img_name}{ext}"
                if mask_path.exists():
                    break
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Normalize mask to 0-255 if needed
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            
            # Binary mask
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Display original
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title(f"Original: {img_name}")
            axes[idx, 0].axis('off')
            
            # Display mask
            axes[idx, 1].imshow(binary_mask, cmap='gray')
            axes[idx, 1].set_title("Mask")
            axes[idx, 1].axis('off')
            
            # Find bounding box of the tool (non-zero mask area)
            coords = cv2.findNonZero(binary_mask)
            if coords is not None:
                x_min, y_min, box_w, box_h = cv2.boundingRect(coords)
                x_max = x_min + box_w
                y_max = y_min + box_h
                
                # Crop image and mask to bounding box
                img_cropped = img[y_min:y_max, x_min:x_max]
                mask_cropped = binary_mask[y_min:y_max, x_min:x_max]
                
                # Create RGBA image from cropped region
                tool_rgba = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2RGBA)
                tool_rgba[:, :, 3] = mask_cropped
                
                # Display extracted tool (cropped)
                axes[idx, 2].imshow(tool_rgba)
                axes[idx, 2].set_title("Tool (Cropped)")
                axes[idx, 2].axis('off')
                
                # Save as PNG
                if save_dir:
                    output_path = save_dir / f"{img_name}_tool.png"
                    cv2.imwrite(
                        str(output_path),
                        cv2.cvtColor(tool_rgba, cv2.COLOR_RGBA2BGRA)
                    )
            else:
                axes[idx, 2].text(0.5, 0.5, 'No tool found', ha='center', va='center')
        else:
            axes[idx, 0].text(0.5, 0.5, 'Mask not found', ha='center', va='center')
            axes[idx, 1].text(0.5, 0.5, 'Mask not found', ha='center', va='center')
            axes[idx, 2].text(0.5, 0.5, 'Mask not found', ha='center', va='center')
    
    plt.tight_layout()
    plt.show()