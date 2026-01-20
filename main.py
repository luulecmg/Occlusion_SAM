"""
Main script for occlusion robustness evaluation.

Usage:
    python main.py --dataset cvc --model medsam --occlusion cutout --ratio 0.2
    python main.py --dataset kvasir --model sam --occlusion none
    python main.py --dataset cvc --model medsam --occlusion cutmix --ratio 0.1 0.2 0.3
    python main.py --dataset kvasir --model medsam --occlusion surgical_tool --ratio 0.2 --tools_dir ./outputs/extracted_tools
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from data import get_dataset
from models import get_model
from occlusion import get_occlusion
from evaluation import compute_metrics
from utils import get_bbox_from_mask, setup_seed, visualize_sample, save_results
from utils.visualization import plot_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SAM/MedSAM robustness to occlusion"
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        choices=['cvc', 'kvasir'],
        help='Dataset to use'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        choices=['sam', 'medsam'],
        help='Model to evaluate'
    )
    
    parser.add_argument(
        '--occlusion', 
        type=str, 
        required=True,
        choices=['none', 'cutout', 'cutmix', 'surgical_tool'],
        help='Occlusion technique'
    )
    
    parser.add_argument(
        '--tools_dir',
        type=str,
        default='./outputs/extracted_tools',
        help='Directory containing extracted surgical tools (required for surgical_tool occlusion)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--ratio', 
        type=float, 
        nargs='+',
        default=[0.0],
        help='Occlusion ratio(s), e.g., --ratio 0.1 0.2 0.3'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='outputs',
        help='Output directory'
    )
    
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=None,
        help='Number of samples to evaluate (None = all)'
    )
    
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='Save visualization for each sample'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--bbox_padding', 
        type=int, 
        default=10,
        help='Bounding box padding'
    )
    
    return parser.parse_args()


def run_evaluation(args):
    """Run the evaluation pipeline."""
    
    # Setup
    setup_seed(args.seed)
    
    # Create output directories with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    results_dir = output_dir / 'results'
    vis_dir = output_dir / 'visualizations'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if args.visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading dataset: {args.dataset}")
    dataset = get_dataset(args.dataset)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = get_model(args.model, device=args.device)
    print(f"Device: {args.device}")
    
    # Determine ratios
    if args.occlusion == 'none':
        ratios = [0.0]
    else:
        ratios = args.ratio
    
    print(f"\nOcclusion: {args.occlusion}")
    print(f"Ratios: {ratios}")
    print(f"{'='*60}\n")
    
    # Run evaluation
    all_results = []
    
    for ratio in ratios:
        # Get occlusion technique
        occlusion_kwargs = {'ratio': ratio}
        if args.occlusion == 'surgical_tool':
            occlusion_kwargs['tools_dir'] = args.tools_dir
        
        occlusion = get_occlusion(args.occlusion, **occlusion_kwargs)
        print(f"\nEvaluating with {occlusion}")
        
        # Limit samples if specified
        num_samples = args.num_samples if args.num_samples else len(dataset)
        num_samples = min(num_samples, len(dataset))
        
        for i in tqdm(range(num_samples), desc=f"Ratio={ratio}"):
            # Get sample
            sample = dataset[i]
            image = sample['image']
            gt_mask = sample['mask']
            sample_id = sample['id']
            
            # Get bounding box from ground truth
            bbox = get_bbox_from_mask(gt_mask, padding=args.bbox_padding)
            
            # Apply occlusion
            occluded = occlusion(image, gt_mask, bbox)
            occluded_image = occluded['image']
            occlusion_mask = occluded['occlusion_mask']
            
            # Run model prediction
            pred_mask = model.predict(occluded_image, bbox)
            
            # Compute metrics
            metrics = compute_metrics(pred_mask, gt_mask)
            
            # Store result
            result = {
                'sample_id': sample_id,
                'dataset': args.dataset,
                'model': args.model,
                'occlusion_type': args.occlusion,
                'occlusion_ratio': ratio,
                **metrics
            }
            all_results.append(result)
            
            # Visualize if requested
            if args.visualize:
                vis_path = vis_dir / f"{sample_id}_{args.occlusion}_{ratio}.png"
                visualize_sample(
                    image=occluded_image,
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                    bbox=bbox,
                    occlusion_mask=occlusion_mask,
                    metrics=metrics,
                    title=f"{sample_id} | {args.occlusion} | ratio={ratio}",
                    save_path=str(vis_path)
                )
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    csv_name = f"{args.dataset}_{args.model}_{args.occlusion}.csv"
    csv_path = results_dir / csv_name
    save_results(all_results, str(csv_path))
    
    # Print summary
    print_summary(df)
    
    # Plot summary
    plot_summary(df, str(results_dir))
    
    return df


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nDataset: {df['dataset'].iloc[0]}")
    print(f"Model: {df['model'].iloc[0]}")
    print(f"Occlusion: {df['occlusion_type'].iloc[0]}")
    print(f"Total samples: {len(df)}")
    
    # Group by occlusion ratio
    print("\n--- Results by Occlusion Ratio ---")
    
    metrics = ['dsc', 'iou', 'hausdorff95', 'assd']
    
    for ratio in sorted(df['occlusion_ratio'].unique()):
        ratio_df = df[df['occlusion_ratio'] == ratio]
        print(f"\nRatio = {ratio}:")
        
        for metric in metrics:
            if metric in ratio_df.columns:
                values = ratio_df[metric].replace([float('inf')], float('nan')).dropna()
                if len(values) > 0:
                    print(f"  {metric.upper():>12}: {values.mean():.4f} ± {values.std():.4f}")
    
    # Overall summary
    print("\n--- Overall ---")
    for metric in metrics:
        if metric in df.columns:
            values = df[metric].replace([float('inf')], float('nan')).dropna()
            if len(values) > 0:
                print(f"  {metric.upper():>12}: {values.mean():.4f} ± {values.std():.4f}")
    
    print("\n" + "="*60)


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == '__main__':
    main()