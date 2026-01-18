"""
Visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
import pandas as pd


def visualize_sample(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    bbox: tuple,
    occlusion_mask: Optional[np.ndarray] = None,
    metrics: Optional[Dict] = None,
    title: str = "",
    save_path: Optional[str] = None
):
    """
    Visualize a single sample result.
    
    Args:
        image: RGB image
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        bbox: Bounding box
        occlusion_mask: Occlusion mask (optional)
        metrics: Dict of metrics (optional)
        title: Title for the plot
        save_path: Path to save figure (optional)
    """
    n_cols = 5 if occlusion_mask is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
    
    # 1. Original image + bbox
    axes[0].imshow(image)
    x_min, y_min, x_max, y_max = bbox
    rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                         fill=False, edgecolor='red', linewidth=2)
    axes[0].add_patch(rect)
    axes[0].set_title('Image + BBox')
    axes[0].axis('off')
    
    # 2. Ground truth
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # 3. Prediction
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # 4. Overlay
    overlay = image.copy().astype(float)
    overlay[gt_mask > 0] = overlay[gt_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    overlay[pred_mask > 0] = overlay[pred_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    axes[3].imshow(overlay.astype(np.uint8))
    axes[3].set_title('Overlay (G=GT, R=Pred)')
    axes[3].axis('off')
    
    # 5. Occlusion mask (if provided)
    if occlusion_mask is not None:
        occ_vis = image.copy()
        occ_vis[occlusion_mask > 0] = [255, 255, 0]  # Yellow
        axes[4].imshow(occ_vis)
        axes[4].set_title('Occlusion Region')
        axes[4].axis('off')
    
    # Add metrics text
    if metrics:
        metrics_text = ' | '.join([f'{k.upper()}: {v:.3f}' for k, v in metrics.items() 
                                   if not np.isinf(v)])
        fig.suptitle(f'{title}\n{metrics_text}', fontsize=10)
    else:
        fig.suptitle(title, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_results(results: list, save_path: str):
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        save_path: Path to save CSV
    """
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")


def plot_summary(df: pd.DataFrame, save_dir: str):
    """
    Plot summary statistics.
    
    Args:
        df: Results DataFrame
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Metrics by occlusion ratio
    if 'occlusion_ratio' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric in ['dsc', 'iou']:
            if metric in df.columns:
                grouped = df.groupby('occlusion_ratio')[metric].agg(['mean', 'std'])
                ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                           label=metric.upper(), marker='o', capsize=5)
        
        ax.set_xlabel('Occlusion Ratio')
        ax.set_ylabel('Score')
        ax.set_title('Performance vs Occlusion Ratio')
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'occlusion_sensitivity.png', dpi=150)
        plt.close()
    
    # 2. Box plot by occlusion technique
    if 'occlusion_type' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for ax, metric in zip(axes, ['dsc', 'iou']):
            if metric in df.columns:
                df.boxplot(column=metric, by='occlusion_type', ax=ax)
                ax.set_title(f'{metric.upper()} by Occlusion Type')
                ax.set_xlabel('Occlusion Type')
                ax.set_ylabel(metric.upper())
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(save_dir / 'metrics_by_occlusion_type.png', dpi=150)
        plt.close()
    
    print(f"Summary plots saved to: {save_dir}")