"""
Visualize model predictions on validation set
"""
import sys
sys.path.append('src')

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from data.hc18_dataset_handler import create_data_loaders
from models.segmentation_models import create_fetal_head_model

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
print("Loading validation data...")
_, val_loader = create_data_loaders(
    organized_dir='data/processed/hc18',
    batch_size=4,
    image_size=(512, 512),
    num_workers=0
)

# Load model
print("Loading best model...")
model = create_fetal_head_model('unet_resnet34', pretrained=False)
checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"Model loaded from epoch {checkpoint['epoch']}")

# Process ALL validation samples
print("Processing all validation samples...")
all_images = []
all_masks_true = []
all_masks_pred = []

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        images = batch['image'].to(device)
        masks_true = batch['mask'].to(device)
        
        # Get predictions
        masks_pred = model(images)
        masks_pred_sigmoid = torch.sigmoid(masks_pred)
        
        all_images.append(images.cpu())
        all_masks_true.append(masks_true.cpu())
        all_masks_pred.append(masks_pred_sigmoid.cpu())

# Concatenate all batches
images = torch.cat(all_images, dim=0)
masks_true = torch.cat(all_masks_true, dim=0)
masks_pred = torch.cat(all_masks_pred, dim=0)

print(f"Total validation samples: {len(images)}")

# Already on CPU after concatenation, no need to move again

# Denormalize images
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Create visualization for first 8 samples
num_samples_viz = min(8, len(images))
fig, axes = plt.subplots(num_samples_viz, 4, figsize=(16, 4*num_samples_viz))

if num_samples_viz == 1:
    axes = axes.reshape(1, -1)

for i in range(num_samples_viz):
    # Denormalize image
    img = images[i].permute(1, 2, 0).numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    # Get masks
    mask_true = masks_true[i].numpy()
    mask_pred = masks_pred[i, 0].numpy()
    mask_pred_binary = (mask_pred > 0.5).astype(np.float32)
    
    # Calculate metrics
    intersection = (mask_pred_binary * mask_true).sum()
    union = mask_pred_binary.sum() + mask_true.sum() - intersection
    iou = intersection / (union + 1e-6)
    
    # Count pixels
    true_pixels = int(mask_true.sum())
    pred_pixels = int(mask_pred_binary.sum())
    
    # Original image
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f'Sample {i+1}: Original Image')
    axes[i, 0].axis('off')
    
    # Ground truth mask
    axes[i, 1].imshow(mask_true, cmap='gray', vmin=0, vmax=1)
    axes[i, 1].set_title(f'Ground Truth\n({true_pixels} pixels)')
    axes[i, 1].axis('off')
    
    # Predicted mask (probability)
    axes[i, 2].imshow(mask_pred, cmap='hot', vmin=0, vmax=1)
    axes[i, 2].set_title(f'Prediction (prob)\nIoU: {iou*100:.1f}%')
    axes[i, 2].axis('off')
    
    # Predicted mask (binary)
    axes[i, 3].imshow(mask_pred_binary, cmap='gray', vmin=0, vmax=1)
    axes[i, 3].set_title(f'Prediction (binary)\n({pred_pixels} pixels)')
    axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig('validation_predictions.png', dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to validation_predictions.png")
plt.close()

# Create overlay visualization for first 8 samples
fig, axes = plt.subplots(2, num_samples_viz, figsize=(4*num_samples_viz, 8))

if num_samples_viz == 1:
    axes = axes.reshape(-1, 1)

for i in range(num_samples_viz):
    # Denormalize image
    img = images[i].permute(1, 2, 0).numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    # Get masks
    mask_true = masks_true[i].numpy()
    mask_pred_binary = (masks_pred[i, 0].numpy() > 0.5).astype(np.float32)
    
    # Overlay ground truth
    img_true_overlay = img.copy()
    img_true_overlay[:, :, 1] = np.maximum(img_true_overlay[:, :, 1], mask_true * 0.7)  # Green
    
    axes[0, i].imshow(img_true_overlay)
    axes[0, i].set_title(f'Sample {i+1}: Ground Truth Overlay')
    axes[0, i].axis('off')
    
    # Overlay prediction
    img_pred_overlay = img.copy()
    img_pred_overlay[:, :, 0] = np.maximum(img_pred_overlay[:, :, 0], mask_pred_binary * 0.7)  # Red
    
    axes[1, i].imshow(img_pred_overlay)
    axes[1, i].set_title(f'Sample {i+1}: Prediction Overlay')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('validation_overlays.png', dpi=150, bbox_inches='tight')
print(f"Saved overlay visualization to validation_overlays.png")
plt.close()

# Ellipse fitting comparison
print(f"\n{'='*60}")
print("ELLIPSE FITTING COMPARISON")
print(f"{'='*60}")

from data.ellipse_fitting import EllipseFitter
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

ellipse_fitter = EllipseFitter()
pixel_size = 0.274  # Average pixel size from HC18 dataset

# Suppress numpy array printing
import sys
import io
old_stdout = sys.stdout

# Create ellipse comparison visualization for first 8 samples
fig, axes = plt.subplots(num_samples_viz, 3, figsize=(15, 5*num_samples_viz))

if num_samples_viz == 1:
    axes = axes.reshape(1, -1)

ellipse_results = []

# Process ALL samples for statistics (but only visualize first 8)
print(f"\nProcessing ellipse fitting for all {len(images)} samples...")
for i in range(len(images)):
    # Denormalize image
    img = images[i].permute(1, 2, 0).numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    # Get masks
    mask_true = masks_true[i].numpy()
    mask_pred = masks_pred[i, 0].numpy()
    mask_pred_binary = (mask_pred > 0.5).astype(np.float32)
    
    # Convert to uint8 for ellipse fitting
    mask_true_uint8 = (mask_true * 255).astype(np.uint8)
    mask_pred_uint8 = (mask_pred_binary * 255).astype(np.uint8)
    
    # Fit ellipses (suppress output)
    sys.stdout = io.StringIO()
    result_true = ellipse_fitter.process_mask(mask_true_uint8, pixel_size=pixel_size)
    result_pred = ellipse_fitter.process_mask(mask_pred_uint8, pixel_size=pixel_size)
    sys.stdout = old_stdout
    
    # Only visualize first num_samples_viz
    if i < num_samples_viz:
        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {i+1}: Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth with ellipse
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(mask_true, cmap='Greens', alpha=0.3)
        
        if result_true['success']:
            ellipse_true = result_true['ellipse_parameters']
            center = (int(ellipse_true['center_x']), int(ellipse_true['center_y']))
            axes_len = (int(ellipse_true['semi_major_axis']), int(ellipse_true['semi_minor_axis']))
            angle = ellipse_true['angle_degrees']
            
            # Draw ellipse
            from matplotlib.patches import Ellipse
            ellipse_patch = Ellipse(center, axes_len[0]*2, axes_len[1]*2, angle=angle,
                                   fill=False, edgecolor='green', linewidth=2)
            axes[i, 1].add_patch(ellipse_patch)
            
            hc_true = result_true['hc_measurements']['head_circumference_mm']
            axes[i, 1].set_title(f'Ground Truth\nHC: {hc_true:.1f} mm')
        else:
            axes[i, 1].set_title(f'Ground Truth\nEllipse fit failed')
        
        axes[i, 1].axis('off')
        
        # Prediction with ellipse
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(mask_pred_binary, cmap='Reds', alpha=0.3)
        
        if result_pred['success']:
            ellipse_pred = result_pred['ellipse_parameters']
            center = (int(ellipse_pred['center_x']), int(ellipse_pred['center_y']))
            axes_len = (int(ellipse_pred['semi_major_axis']), int(ellipse_pred['semi_minor_axis']))
            angle = ellipse_pred['angle_degrees']
            
            # Draw ellipse
            ellipse_patch = Ellipse(center, axes_len[0]*2, axes_len[1]*2, angle=angle,
                                   fill=False, edgecolor='red', linewidth=2)
            axes[i, 2].add_patch(ellipse_patch)
            
            hc_pred = result_pred['hc_measurements']['head_circumference_mm']
            axes[i, 2].set_title(f'Prediction\nHC: {hc_pred:.1f} mm')
        else:
            axes[i, 2].set_title(f'Prediction\nEllipse fit failed')
        
        axes[i, 2].axis('off')
    
    # Store results
    true_points = len(result_true.get('contour', [])) if result_true.get('contour') is not None else 0
    pred_points = len(result_pred.get('contour', [])) if result_pred.get('contour') is not None else 0
    
    ellipse_results.append({
        'sample': i+1,
        'true_success': result_true['success'],
        'pred_success': result_pred['success'],
        'true_hc': result_true['hc_measurements']['head_circumference_mm'] if result_true['success'] else None,
        'pred_hc': result_pred['hc_measurements']['head_circumference_mm'] if result_pred['success'] else None,
        'true_contour_points': true_points,
        'pred_contour_points': pred_points
    })

plt.tight_layout()
plt.savefig('ellipse_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved ellipse comparison to ellipse_comparison.png")

# Restore stdout
sys.stdout = old_stdout

# Print summary statistics for ALL samples
print(f"\n{'='*60}")
print(f"SUMMARY STATISTICS (All {len(ellipse_results)} validation samples)")
print(f"{'='*60}")

successful_both = [r for r in ellipse_results if r['true_success'] and r['pred_success']]
if successful_both:
    hc_errors = [abs(r['pred_hc'] - r['true_hc']) for r in successful_both]
    hc_errors_pct = [(abs(r['pred_hc'] - r['true_hc']) / r['true_hc']) * 100 for r in successful_both]
    
    print(f"\nEllipse Fitting Success Rate:")
    print(f"  Ground truth: {sum(1 for r in ellipse_results if r['true_success'])}/{len(ellipse_results)} ({sum(1 for r in ellipse_results if r['true_success'])/len(ellipse_results)*100:.1f}%)")
    print(f"  Predictions: {sum(1 for r in ellipse_results if r['pred_success'])}/{len(ellipse_results)} ({sum(1 for r in ellipse_results if r['pred_success'])/len(ellipse_results)*100:.1f}%)")
    print(f"  Both successful: {len(successful_both)}/{len(ellipse_results)} ({len(successful_both)/len(ellipse_results)*100:.1f}%)")
    
    print(f"\nHead Circumference Error (for {len(successful_both)} successful pairs):")
    print(f"  Mean error: {np.mean(hc_errors):.2f} mm ({np.mean(hc_errors_pct):.2f}%)")
    print(f"  Median error: {np.median(hc_errors):.2f} mm ({np.median(hc_errors_pct):.2f}%)")
    print(f"  Std dev: {np.std(hc_errors):.2f} mm")
    print(f"  Min error: {np.min(hc_errors):.2f} mm ({np.min(hc_errors_pct):.2f}%)")
    print(f"  Max error: {np.max(hc_errors):.2f} mm ({np.max(hc_errors_pct):.2f}%)")
    
    # Count samples within different error thresholds
    within_5mm = sum(1 for e in hc_errors if e <= 5.0)
    within_10mm = sum(1 for e in hc_errors if e <= 10.0)
    within_15mm = sum(1 for e in hc_errors if e <= 15.0)
    
    print(f"\nError Distribution:")
    print(f"  Within 5mm: {within_5mm}/{len(successful_both)} ({within_5mm/len(successful_both)*100:.1f}%)")
    print(f"  Within 10mm: {within_10mm}/{len(successful_both)} ({within_10mm/len(successful_both)*100:.1f}%)")
    print(f"  Within 15mm: {within_15mm}/{len(successful_both)} ({within_15mm/len(successful_both)*100:.1f}%)")

# Print detailed statistics for first 10 samples
print(f"\n{'='*60}")
print("DETAILED STATISTICS (First 10 samples)")
print(f"{'='*60}")

for i, result in enumerate(ellipse_results[:10]):
    mask_true = masks_true[i].numpy()
    mask_pred = masks_pred[i, 0].numpy()
    mask_pred_binary = (mask_pred > 0.5).astype(np.float32)
    
    true_pixels = int(mask_true.sum())
    pred_pixels = int(mask_pred_binary.sum())
    
    intersection = (mask_pred_binary * mask_true).sum()
    union = mask_pred_binary.sum() + mask_true.sum() - intersection
    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (mask_pred_binary.sum() + mask_true.sum() + 1e-6)
    
    print(f"\nSample {result['sample']}:")
    print(f"  Segmentation Metrics:")
    print(f"    Ground truth pixels: {true_pixels}")
    print(f"    Predicted pixels: {pred_pixels}")
    print(f"    IoU: {iou*100:.2f}%")
    print(f"    Dice: {dice*100:.2f}%")
    
    print(f"  Ellipse Fitting:")
    print(f"    Ground truth: {'✓' if result['true_success'] else '✗'} ({result['true_contour_points']} contour points)")
    print(f"    Prediction: {'✓' if result['pred_success'] else '✗'} ({result['pred_contour_points']} contour points)")
    
    if result['true_success'] and result['pred_success']:
        hc_error = abs(result['pred_hc'] - result['true_hc'])
        hc_error_pct = (hc_error / result['true_hc']) * 100
        print(f"  Head Circumference:")
        print(f"    Ground truth: {result['true_hc']:.2f} mm")
        print(f"    Prediction: {result['pred_hc']:.2f} mm")
        print(f"    Error: {hc_error:.2f} mm ({hc_error_pct:.1f}%)")
    elif result['true_success']:
        print(f"  Head Circumference:")
        print(f"    Ground truth: {result['true_hc']:.2f} mm")
        print(f"    Prediction: Failed to fit ellipse")
    else:
        print(f"  Head Circumference: Both failed to fit ellipse")

print(f"\n{'='*60}")
