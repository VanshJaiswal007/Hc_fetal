"""
Test the model inference pipeline
"""
import sys
import os
sys.path.append('..')

from utils.model_inference import ModelInference
from PIL import Image
import numpy as np

print("=" * 60)
print("Testing Model Inference Pipeline")
print("=" * 60)

# Test with a sample image - try multiple possible paths
possible_paths = [
    '../hc18_dataset/training_set/training_set/1_HC.png',
    'hc18_dataset/training_set/training_set/1_HC.png',
    '../data/processed/hc18/images/train/1_HC.png',
    'data/processed/hc18/images/train/1_HC.png'
]

test_image_path = None
for path in possible_paths:
    if os.path.exists(path):
        test_image_path = path
        break

if test_image_path is None:
    print(f"Error: Test image not found. Tried:")
    for path in possible_paths:
        print(f"  - {path}")
    print("\nPlease provide a valid image path or run from the correct directory.")
    sys.exit(1)

pixel_spacing = 0.0691358041432  # From CSV

print(f"\nTest image: {test_image_path}")
print(f"Pixel spacing: {pixel_spacing} mm/pixel")

# Initialize model inference
print("\nInitializing model...")
try:
    model_inference = ModelInference(
        model_path='../checkpoints/best_model.pth',
        device=None  # Auto-detect
    )
    print("✓ Model initialized")
except Exception as e:
    print(f"✗ Error initializing model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Run prediction
print("\nRunning prediction...")
try:
    result = model_inference.predict(test_image_path, pixel_spacing)
    print("✓ Prediction complete")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nEllipse fitting: {'✓ Success' if result['ellipse_fitting_success'] else '✗ Failed'}")
    
    if result['head_circumference_mm']:
        print(f"\nHead Circumference: {result['head_circumference_mm']:.2f} mm")
        
        if result['hc_measurements']:
            hc = result['hc_measurements']
            print(f"OFD (Occipitofrontal Diameter): {hc['occipitofrontal_diameter_mm']:.2f} mm")
            print(f"BPD (Biparietal Diameter): {hc['biparietal_diameter_mm']:.2f} mm")
            print(f"Aspect Ratio: {hc['aspect_ratio']:.2f}")
            print(f"Area: {hc['area_mm2']:.2f} mm²")
    else:
        print("\n⚠ Could not calculate head circumference")
    
    # Save visualization
    print("\nSaving visualization...")
    viz_path = 'test_prediction.png'
    model_inference.save_visualization(test_image_path, result, viz_path)
    print(f"✓ Saved to: {viz_path}")
    
except Exception as e:
    print(f"✗ Error during prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✨ Test complete!")
print("=" * 60)
