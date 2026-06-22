"""
Test webapp visualization
"""
import sys
sys.path.append('webapp')
sys.path.append('src')

from utils.model_inference import ModelInference
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Test image
test_image_path = 'data/processed/hc18/images/test/100_HC.png'

print("Testing webapp visualization...")
model_inference = ModelInference(model_path='checkpoints/best_model.pth')

# Run prediction
result = model_inference.predict(test_image_path, pixel_spacing=0.274)

print(f"\nPrediction Results:")
print(f"  Mask shape: {result['mask'].shape}")
print(f"  Positive pixels: {result['mask'].sum()}")
print(f"  HC: {result['head_circumference_mm']:.2f} mm")
print(f"  Ellipse fitting: {result['ellipse_fitting_success']}")

# Save visualization
output_path = 'test_webapp_viz_output.png'
model_inference.save_visualization(test_image_path, result, output_path)
print(f"\nSaved visualization to {output_path}")

# Load and display
viz_img = cv2.imread(output_path)
viz_img = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)

# Also load original for comparison
orig_img = cv2.imread(test_image_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

# Create comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(orig_img)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(result['mask'], cmap='gray')
axes[1].set_title(f'Predicted Mask\n{result["mask"].sum():.0f} pixels')
axes[1].axis('off')

axes[2].imshow(viz_img)
axes[2].set_title(f'Webapp Visualization\nHC: {result["head_circumference_mm"]:.1f} mm')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('webapp_visualization_test.png', dpi=150)
print(f"Saved comparison to webapp_visualization_test.png")

# Check if mask aligns with image
print(f"\nImage shape: {orig_img.shape}")
print(f"Mask shape: {result['mask'].shape}")
print(f"Shapes match: {result['mask'].shape == orig_img.shape[:2]}")
