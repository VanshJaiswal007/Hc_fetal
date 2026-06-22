"""
Compare predictions between visualization script method and webapp method
"""
import sys
sys.path.append('src')
sys.path.append('webapp')

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Load model
from models.segmentation_models import create_fetal_head_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = create_fetal_head_model('unetplusplus_resnet34', pretrained=False)
checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Test image
test_image_path = 'data/processed/hc18/images/test/100_HC.png'

print("\n" + "="*60)
print("PREDICTION COMPARISON")
print("="*60)

# Method 1: Visualization script method (using albumentations)
print("\n1. Visualization Script Method:")
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

img_cv2 = cv2.imread(test_image_path)
img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
original_size = (img_cv2.shape[1], img_cv2.shape[0])
print(f"  Original size: {original_size}")

transformed = transform(image=img_cv2)
img_tensor = transformed['image'].unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    pred_prob = torch.sigmoid(output)
    pred_mask = (pred_prob > 0.5).cpu().numpy()[0, 0]

print(f"  Prediction at 512x512: {pred_mask.sum()} positive pixels")

# Resize back to original (this is what webapp does)
pred_mask_resized = cv2.resize(
    pred_prob.cpu().numpy()[0, 0],
    original_size,
    interpolation=cv2.INTER_LINEAR
)
pred_mask_resized_binary = (pred_mask_resized > 0.5).astype(np.float32)
print(f"  After resize to {original_size}: {pred_mask_resized_binary.sum()} positive pixels")

# Method 2: Webapp method (current implementation)
print("\n2. Webapp Method:")
from utils.model_inference import ModelInference

model_inference = ModelInference(model_path='checkpoints/best_model.pth', device=device)
result = model_inference.predict(test_image_path, pixel_spacing=0.274)

print(f"  Prediction at original size: {result['mask'].sum()} positive pixels")
print(f"  HC: {result['head_circumference_mm']}")

# Compare
print("\n3. Comparison:")
diff = np.abs(pred_mask_resized_binary - result['mask'])
print(f"  Difference in positive pixels: {np.abs(pred_mask_resized_binary.sum() - result['mask'].sum())}")
print(f"  Pixels that differ: {diff.sum()}")
print(f"  Percentage difference: {(diff.sum() / result['mask'].size) * 100:.2f}%")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original image
axes[0, 0].imshow(img_cv2)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Visualization script - at 512x512
axes[0, 1].imshow(pred_mask, cmap='gray')
axes[0, 1].set_title(f'Viz Script @ 512x512\n{pred_mask.sum():.0f} pixels')
axes[0, 1].axis('off')

# Visualization script - resized back
axes[0, 2].imshow(pred_mask_resized_binary, cmap='gray')
axes[0, 2].set_title(f'Viz Script Resized\n{pred_mask_resized_binary.sum():.0f} pixels')
axes[0, 2].axis('off')

# Webapp result
axes[1, 0].imshow(result['mask'], cmap='gray')
axes[1, 0].set_title(f'Webapp Result\n{result["mask"].sum():.0f} pixels')
axes[1, 0].axis('off')

# Difference
axes[1, 1].imshow(diff, cmap='hot')
axes[1, 1].set_title(f'Difference\n{diff.sum():.0f} pixels differ')
axes[1, 1].axis('off')

# Overlay comparison
overlay = img_cv2.copy()
# Green = visualization script
overlay[:, :, 1] = np.maximum(overlay[:, :, 1], pred_mask_resized_binary * 150)
# Red = webapp
overlay[:, :, 0] = np.maximum(overlay[:, :, 0], result['mask'] * 150)
axes[1, 2].imshow(overlay)
axes[1, 2].set_title('Overlay (Green=Viz, Red=Webapp)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('prediction_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved comparison to prediction_comparison.png")

print("\n" + "="*60)
