# models/sam_wrapper.py
"""
SAM and MedSAM model wrapper for inference.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class SAMWrapper:
    """Wrapper for SAM/MedSAM models."""
    
    def __init__(
        self, 
        checkpoint_path: str,
        model_type: str = "vit_b",
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = 1024
        
        # Load model
        self.model = self._load_model(checkpoint_path, model_type)
        self.model.eval()
        
        # Normalization parameters
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(self.device)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(self.device)
    
    def _load_model(self, checkpoint_path: str, model_type: str):
        """Load SAM/MedSAM model."""
        from segment_anything import sam_model_registry
        
        # Use map_location to load CUDA checkpoints on CPU if needed
        if str(self.device) == 'cpu':
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model = sam_model_registry[model_type](checkpoint=None)
            model.load_state_dict(checkpoint)
        else:
            model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        
        model = model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def predict(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Predict segmentation mask.
        
        Args:
            image: RGB image (H, W, 3), values 0-255
            bbox: Bounding box (x_min, y_min, x_max, y_max)
        
        Returns:
            Binary mask (H, W)
        """
        H, W = image.shape[:2]
        
        # Preprocess image
        img_tensor = self._preprocess(image)
        
        # Scale bbox
        scale_x = self.image_size / W
        scale_y = self.image_size / H
        bbox_scaled = [
            bbox[0] * scale_x,
            bbox[1] * scale_y,
            bbox[2] * scale_x,
            bbox[3] * scale_y
        ]
        bbox_tensor = torch.tensor([bbox_scaled], device=self.device).unsqueeze(0)
        
        # Get image embedding
        image_embedding = self.model.image_encoder(img_tensor)
        
        # Get prompt embedding
        sparse_emb, dense_emb = self.model.prompt_encoder(
            points=None,
            boxes=bbox_tensor,
            masks=None
        )
        
        # Decode mask
        low_res_mask, _ = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False
        )
        
        # Postprocess
        mask = self._postprocess(low_res_mask, (H, W))
        
        return mask
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize to 1024x1024
        img_resized = F.interpolate(
            torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).to(self.device)
        
        # Normalize
        img_normalized = (img_resized - self.pixel_mean) / self.pixel_std
        
        return img_normalized
    
    def _postprocess(
        self, 
        low_res_mask: torch.Tensor, 
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """Postprocess mask to original size."""
        # Upsample to 1024
        mask = F.interpolate(
            low_res_mask,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Resize to original
        mask = F.interpolate(
            mask,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Threshold
        mask = (mask > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        
        return mask


def get_model(name: str, device: str = "cuda") -> SAMWrapper:
    """
    Factory function to get model by name.
    
    Args:
        name: 'sam' or 'medsam'
        device: 'cuda' or 'mps' or 'cpu'
    
    Returns:
        SAMWrapper instance
    """
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    models = {
        'sam': {
            'checkpoint': 'checkpoints/sam_vit_b.pth',
            'type': 'vit_b'
        },
        'medsam': {
            'checkpoint': 'checkpoints/medsam_vit_b.pth',
            'type': 'vit_b'
        }
    }
    
    if name.lower() not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    
    config = models[name.lower()]
    
    return SAMWrapper(
        checkpoint_path=config['checkpoint'],
        model_type=config['type'],
        device=device
    )