"""
Visualize ALL validation predictions with ellipse fitting
"""
import sys
sys.path.append('src')

import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from data.hc18_dataset_handler import create_data_loaders
from models.segmentation_models import create_fetal_head_model
from data.ellipse_fitting import EllipseFitter

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
print("Loading validation data...")
_, val_loader = create_data_loaders(
    organized_dir='data/processed/hc18',
    batch_size=8,
    image_size=(512, 512),
    num_workers=0
)

# Load model
print("Loading best model...")
model = create_fetal_head_model('unet_resnet34', pretrained=False)
checkpoint = torch.load('checkpoints/best_model.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print(f"Model loaded from epoch {checkpoint['epoch']}")

# Initialize ellipse fitter
ellipse_fitter = EllipseFitter()
pixel_size = 0.274  # Average pixel size from HC18 dataset

# Process ALL validation samples
print(f"\nProcessing all validation samples...")
all_results = []

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        images = batch['image'].to(device)
        masks_true = batch['mask'].to(device)
        
        # Get predictions
        masks_pred = model(images)
        masks_pred_sigmoid = torch.sigmoid(masks_pred)
        
        # Move to CPU
        images_cpu = images.cpu()
        masks_true_cpu = masks_true.cpu()
        masks_pred_cpu = masks_pred_sigmoid.cpu()
        
        # Process each sample in batch
        for i in range(len(images)):
            mask_true = masks_true_cpu[i].numpy()
            mask_pred = masks_pred_cpu[i, 0].numpy()
            mask_pred_binary = (mask_pred > 0.5).astype(np.float32)
            
            # Convert to uint8 for ellipse fitting
            mask_true_uint8 = (mask_true * 255).astype(np.uint8)
            mask_pred_uint8 = (mask_pred_binary * 255).astype(np.uint8)
            
            # Fit ellipses
            result_true = ellipse_fitter.process_mask(mask_true_uint8, pixel_size=pixel_size)
            result_pred = ellipse_fitter.process_mask(mask_pred_uint8, pixel_size=pixel_size)
            
            # Calculate segmentation metrics
            intersection = (mask_pred_binary * mask_true).sum()
            union = mask_pred_binary.sum() + mask_true.sum() - intersection
            iou = intersection / (union + 1e-6)
            dice = (2 * intersection) / (mask_pred_binary.sum() + mask_true.sum() + 1e-6)
            
            # Store results
            all_results.append({
                'sample_idx': batch_idx * val_loader.batch_size + i + 1,
                'iou': iou * 100,
                'dice': dice * 100,
                'true_pixels': int(mask_true.sum()),
                'pred_pixels': int(mask_pred_binary.sum()),
                'true_success': result_true['success'],
                'pred_success': result_pred['success'],
                'true_hc': result_true['hc_measurements']['head_circumference_mm'] if result_true['success'] else None,
                'pred_hc': result_pred['hc_measurements']['head_circumference_mm'] if result_pred['success'] else None,
            })

print(f"Processed {len(all_results)} validation samples")

# Print summary statistics
print(f"\n{'='*70}")
print(f"VALIDATION RESULTS SUMMARY ({len(all_results)} samples)")
print(f"{'='*70}")

# Segmentation metrics
ious = [r['iou'] for r in all_results]
dices = [r['dice'] for r in all_results]

print(f"\nSegmentation Metrics:")
print(f"  Mean IoU: {np.mean(ious):.2f}% (std: {np.std(ious):.2f}%)")
print(f"  Median IoU: {np.median(ious):.2f}%")
print(f"  Mean Dice: {np.mean(dices):.2f}% (std: {np.std(dices):.2f}%)")

# Ellipse fitting success
true_success = sum(1 for r in all_results if r['true_success'])
pred_success = sum(1 for r in all_results if r['pred_success'])
both_success = sum(1 for r in all_results if r['true_success'] and r['pred_success'])

print(f"\nEllipse Fitting Success:")
print(f"  Ground truth: {true_success}/{len(all_results)} ({true_success/len(all_results)*100:.1f}%)")
print(f"  Predictions: {pred_success}/{len(all_results)} ({pred_success/len(all_results)*100:.1f}%)")
print(f"  Both successful: {both_success}/{len(all_results)} ({both_success/len(all_results)*100:.1f}%)")

# HC measurement errors (only for successful pairs)
successful_pairs = [r for r in all_results if r['true_success'] and r['pred_success']]

if successful_pairs:
    hc_errors = [abs(r['pred_hc'] - r['true_hc']) for r in successful_pairs]
    hc_errors_pct = [(abs(r['pred_hc'] - r['true_hc']) / r['true_hc']) * 100 for r in successful_pairs]
    
    print(f"\nHead Circumference Errors ({len(successful_pairs)} successful pairs):")
    print(f"  Mean error: {np.mean(hc_errors):.2f} mm ({np.mean(hc_errors_pct):.2f}%)")
    print(f"  Median error: {np.median(hc_errors):.2f} mm ({np.median(hc_errors_pct):.2f}%)")
    print(f"  Std dev: {np.std(hc_errors):.2f} mm")
    print(f"  Min error: {np.min(hc_errors):.2f} mm ({np.min(hc_errors_pct):.2f}%)")
    print(f"  Max error: {np.max(hc_errors):.2f} mm ({np.max(hc_errors_pct):.2f}%)")
    
    # Error distribution
    within_5mm = sum(1 for e in hc_errors if e <= 5.0)
    within_10mm = sum(1 for e in hc_errors if e <= 10.0)
    within_15mm = sum(1 for e in hc_errors if e <= 15.0)
    
    print(f"\n  Error Distribution:")
    print(f"    Within 5mm: {within_5mm}/{len(successful_pairs)} ({within_5mm/len(successful_pairs)*100:.1f}%)")
    print(f"    Within 10mm: {within_10mm}/{len(successful_pairs)} ({within_10mm/len(successful_pairs)*100:.1f}%)")
    print(f"    Within 15mm: {within_15mm}/{len(successful_pairs)} ({within_15mm/len(successful_pairs)*100:.1f}%)")

# Show best and worst predictions
print(f"\n{'='*70}")
print("BEST PREDICTIONS (by HC error):")
print(f"{'='*70}")
best_5 = sorted(successful_pairs, key=lambda r: abs(r['pred_hc'] - r['true_hc']))[:5]
for i, r in enumerate(best_5, 1):
    error = abs(r['pred_hc'] - r['true_hc'])
    error_pct = (error / r['true_hc']) * 100
    print(f"{i}. Sample {r['sample_idx']}: True={r['true_hc']:.1f}mm, Pred={r['pred_hc']:.1f}mm, Error={error:.2f}mm ({error_pct:.1f}%), IoU={r['iou']:.1f}%")

print(f"\n{'='*70}")
print("WORST PREDICTIONS (by HC error):")
print(f"{'='*70}")
worst_5 = sorted(successful_pairs, key=lambda r: abs(r['pred_hc'] - r['true_hc']), reverse=True)[:5]
for i, r in enumerate(worst_5, 1):
    error = abs(r['pred_hc'] - r['true_hc'])
    error_pct = (error / r['true_hc']) * 100
    print(f"{i}. Sample {r['sample_idx']}: True={r['true_hc']:.1f}mm, Pred={r['pred_hc']:.1f}mm, Error={error:.2f}mm ({error_pct:.1f}%), IoU={r['iou']:.1f}%")

print(f"\n{'='*70}")
