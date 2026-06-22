"""
Compare the two visualizations to see what's different
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load both visualizations
viz1 = cv2.imread('test_webapp_viz_output.png')
viz1 = cv2.cvtColor(viz1, cv2.COLOR_BGR2RGB)

viz2 = cv2.imread('webapp/test_prediction.png')
viz2 = cv2.cvtColor(viz2, cv2.COLOR_BGR2RGB)

# Load original images
img1 = cv2.imread('data/processed/hc18/images/test/100_HC.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('hc18_dataset/training_set/training_set/1_HC.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

print("Visualization 1 (test_webapp_viz_output.png):")
print(f"  Shape: {viz1.shape}")
print(f"  From image: data/processed/hc18/images/test/100_HC.png")

print("\nVisualization 2 (test_prediction.png):")
print(f"  Shape: {viz2.shape}")
print(f"  From image: hc18_dataset/training_set/training_set/1_HC.png")

# Create comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Good visualization
axes[0, 0].imshow(img1)
axes[0, 0].set_title('Original: 100_HC.png')
axes[0, 0].axis('off')

axes[0, 1].imshow(viz1)
axes[0, 1].set_title('✓ GOOD Visualization\n(test_webapp_viz_output.png)')
axes[0, 1].axis('off')

# Extract just the red mask from viz1
mask1_red = (viz1[:, :, 0] > 100) & (viz1[:, :, 1] < 100)
axes[0, 2].imshow(mask1_red, cmap='gray')
axes[0, 2].set_title(f'Extracted Mask\n{mask1_red.sum()} pixels')
axes[0, 2].axis('off')

# Row 2: Bad visualization
axes[1, 0].imshow(img2)
axes[1, 0].set_title('Original: 1_HC.png')
axes[1, 0].axis('off')

axes[1, 1].imshow(viz2)
axes[1, 1].set_title('✗ BAD Visualization?\n(test_prediction.png)')
axes[1, 1].axis('off')

# Extract just the red mask from viz2
mask2_red = (viz2[:, :, 0] > 100) & (viz2[:, :, 1] < 100)
axes[1, 2].imshow(mask2_red, cmap='gray')
axes[1, 2].set_title(f'Extracted Mask\n{mask2_red.sum()} pixels')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('visualization_comparison.png', dpi=150)
print("\nSaved comparison to visualization_comparison.png")

# Check if the masks look reasonable
print(f"\nMask 1 coverage: {mask1_red.sum() / mask1_red.size * 100:.2f}%")
print(f"Mask 2 coverage: {mask2_red.sum() / mask2_red.size * 100:.2f}%")
