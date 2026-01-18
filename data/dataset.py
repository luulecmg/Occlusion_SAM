import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class BaseDataset:
    """Base dataset class."""
    
    def __init__(self, root: str, image_dir: str, mask_dir: str):
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.mask_dir = self.root / mask_dir
        
        # Get all samples
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """Load all image-mask pairs."""
        samples = []
        
        # Supported extensions
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.bmp']
        
        for img_path in self.image_dir.iterdir():
            if img_path.suffix.lower() not in extensions:
                continue
            
            # Find corresponding mask
            mask_path = self._find_mask(img_path.stem)
            
            if mask_path is not None:
                samples.append({
                    'id': img_path.stem,
                    'image_path': str(img_path),
                    'mask_path': str(mask_path)
                })
        
        # Sort by id
        samples = sorted(samples, key=lambda x: x['id'])
        
        return samples
    
    def _find_mask(self, image_stem: str) -> Optional[Path]:
        """Find mask file for given image."""
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.bmp']
        
        for ext in extensions:
            mask_path = self.mask_dir / f"{image_stem}{ext}"
            if mask_path.exists():
                return mask_path
        
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample."""
        sample = self.samples[idx]
        
        # Load image (RGB)
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (binary)
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        
        return {
            'id': sample['id'],
            'image': image,
            'mask': mask
        }


class CVCDataset(BaseDataset):
    """CVC-ClinicDB Dataset."""
    
    def __init__(self, root: str = "dataset/CVC-ClinicDB/PNG"):
        super().__init__(
            root=root,
            image_dir="Original",
            mask_dir="Ground_Truth"
        )
        self.name = "CVC-ClinicDB"


class KvasirDataset(BaseDataset):
    """Kvasir-SEG Dataset."""
    
    def __init__(self, root: str = "dataset/Kvasir-SEG"):
        super().__init__(
            root=root,
            image_dir="images",
            mask_dir="masks"
        )
        self.name = "Kvasir-SEG"


def get_dataset(name: str) -> BaseDataset:
    """
    Factory function to get dataset by name.
    
    Args:
        name: 'cvc' or 'kvasir'
    
    Returns:
        Dataset instance
    """
    datasets = {
        'cvc': CVCDataset,
        'kvasir': KvasirDataset,
    }
    
    if name.lower() not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")
    
    return datasets[name.lower()]()