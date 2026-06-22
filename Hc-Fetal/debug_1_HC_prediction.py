"""
Debug prediction on 1_HC.png to see what's happening
"""
import sys
sys.path.append('webapp')
sys.path.append('src')

from utils.model_inference import ModelInference
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Test image - same as test_inference.py uses
test_image_path = 'hc18_dataset/training_set/training_set/1_HC.png'
pixel_spacing = 0.0691358041432

print("Testing prediction on 1_HC.png...")
print(f"Pixel spacing: {pixel_spacing} mm/pixel")

# Load original image
img = cv2.imread(test_image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Image shape: {img_rgb.shape}")

# Run prediction
model_inference = ModelInference(model_path='checkpoints/best_model.pth')
result = model_inference.predict(test_image_path, pixel_spacing)

print(f"\nPrediction Results:")
print(f"  Mask shape: {result['mask'].shape}")
print(f"  Positive pixels: {result['mask'].sum()}")
print(f"  Mask coverage: {result['mask'].sum() / result['mask'].size * 100:.2f}%")
print(f"  HC: {result['head_circumference_mm']:.2f} mm" if result['head_circumference_mm'] else "  HC: Failed")
print(f"  Ellipse fitting: {result['ellipse_fitting_success']}")

# Create detailed visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Prediction details
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(result['mask'], cmap='gray')
axes[0, 1].set_title(f'Predicted Mask\n{result["mask"].sum():.0f} pixels')
axes[0, 1].axis('off')

# Overlay mask on image
overlay1 = img_rgb.copy()
mask_binary = (result['mask'] > 0.5).astype(np.uint8)
overlay1[mask_binary > 0, 0] = 255  # Red channel
axes[0, 2].imshow(overlay1)
axes[0, 2].set_title('Simple Overlay')
axes[0, 2].axis('off')

# Row 2: Using save_visualization method
output_path = 'debug_1_HC_viz.png'
model_inference.save_visualization(test_image_path, result, output_path)
viz_img = cv2.imread(output_path)
viz_img = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)

axes[1, 0].imshow(viz_img)
axes[1, 0].set_title(f'save_visualization Output\nHC: {result["head_circumference_mm"]:.1f} mm' if result['head_circumference_mm'] else 'save_visualization Output')
axes[1, 0].axis('off')

# Compare with test_prediction.png
test_pred_img = cv2.imread('webapp/test_prediction.png')
test_pred_img = cv2.cvtColor(test_pred_img, cv2.COLOR_BGR2RGB)
axes[1, 1].imshow(test_pred_img)
axes[1, 1].set_title('test_prediction.png\n(from test_inference.py)')
axes[1, 1].axis('off')

# Difference
diff = np.abs(viz_img.astype(float) - test_pred_img.astype(float)).mean(axis=2)
axes[1, 2].imshow(diff, cmap='hot')
axes[1, 2].set_title(f'Difference\nMax: {diff.max():.1f}')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('debug_1_HC_comparison.png', dpi=150)
print(f"\nSaved debug visualization to debug_1_HC_comparison.png")

# Check if ellipse parameters are reasonable
if result['ellipse_params']:
    print(f"\nEllipse Parameters:")
    print(f"  Center: ({result['ellipse_params']['center_x']:.1f}, {result['ellipse_params']['center_y']:.1f})")
    print(f"  Semi-major axis: {result['ellipse_params']['semi_major_axis']:.1f} pixels")
    print(f"  Semi-minor axis: {result['ellipse_params']['semi_minor_axis']:.1f} pixels")
    print(f"  Angle: {result['ellipse_params']['angle_degrees']:.1f}°")
