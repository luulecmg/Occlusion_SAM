"""
Segmentation evaluation metrics.
"""

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from typing import Dict


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice Similarity Coefficient (DSC)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = np.logical_and(pred, gt).sum()
    
    if pred.sum() + gt.sum() == 0:
        return 1.0
    
    return (2.0 * intersection) / (pred.sum() + gt.sum())


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over Union (IoU)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        return 1.0
    
    return intersection / union


def compute_hausdorff(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hausdorff Distance (HD)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.inf
    
    pred_points = _get_boundary_points(pred)
    gt_points = _get_boundary_points(gt)
    
    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.inf
    
    hd1 = directed_hausdorff(pred_points, gt_points)[0]
    hd2 = directed_hausdorff(gt_points, pred_points)[0]
    
    return max(hd1, hd2)


def compute_hausdorff95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th percentile Hausdorff Distance (HD95)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.inf
    
    pred_points = _get_boundary_points(pred)
    gt_points = _get_boundary_points(gt)
    
    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.inf
    
    # Compute distances
    tree_gt = cKDTree(gt_points)
    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    
    tree_pred = cKDTree(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points)
    
    all_dist = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    
    return np.percentile(all_dist, 95)


def compute_assd(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average Symmetric Surface Distance (ASSD)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.inf
    
    pred_points = _get_boundary_points(pred)
    gt_points = _get_boundary_points(gt)
    
    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.inf
    
    tree_gt = cKDTree(gt_points)
    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    
    tree_pred = cKDTree(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points)
    
    return (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) / 2


def compute_precision(pred: np.ndarray, gt: np.ndarray) -> float:
    """Precision."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)


def compute_recall(pred: np.ndarray, gt: np.ndarray) -> float:
    """Recall (Sensitivity)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)


def _get_boundary_points(mask: np.ndarray) -> np.ndarray:
    """Extract boundary points from binary mask."""
    if mask.sum() == 0:
        return np.array([]).reshape(0, 2)
    
    eroded = binary_erosion(mask)
    boundary = mask ^ eroded
    points = np.array(np.where(boundary)).T
    
    if len(points) == 0:
        points = np.array(np.where(mask)).T
    
    return points


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Compute all segmentation metrics.
    
    Args:
        pred: Predicted binary mask (H, W)
        gt: Ground truth binary mask (H, W)
    
    Returns:
        Dictionary of metric names and values
    """
    return {
        'dsc': compute_dice(pred, gt),
        'iou': compute_iou(pred, gt),
        'hausdorff': compute_hausdorff(pred, gt),
        'hausdorff95': compute_hausdorff95(pred, gt),
        'assd': compute_assd(pred, gt),
        'precision': compute_precision(pred, gt),
        'recall': compute_recall(pred, gt)
    }


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Alias for compute_all_metrics."""
    return compute_all_metrics(pred, gt)