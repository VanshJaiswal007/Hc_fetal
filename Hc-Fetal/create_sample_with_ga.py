"""
Create a sample visualization with gestational age calculation
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from webapp.utils.gestational_age_calculator import GestationalAgeCalculator

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Create a sample ultrasound-like image (grayscale with circular fetal head)
img = np.zeros((512, 512))
y, x = np.ogrid[:512, :512]
center_y, center_x = 256, 256
radius = 120

# Create circular head region
mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
img[mask] = 0.6

# Add some texture
noise = np.random.normal(0, 0.1, (512, 512))
img = np.clip(img + noise, 0, 1)

# Display image
ax.imshow(img, cmap='gray')

# Draw ellipse for head circumference
ellipse = patches.Ellipse(
    (center_x, center_y), 
    width=radius*2, 
    height=radius*2.1,
    fill=False, 
    edgecolor='cyan', 
    linewidth=2,
    label='Predicted HC'
)
ax.add_patch(ellipse)

# Sample measurements
hc_mm = 224.5  # Example HC at ~24 weeks
pixel_spacing = 0.274

# Calculate gestational age
ga_calc = GestationalAgeCalculator()
ga_result = ga_calc.calculate_ga_from_hc(hc_mm)

# Add measurement text
info_text = f"""Head Circumference: {hc_mm:.1f} mm

Gestational Age: {ga_result['formatted']}
({ga_result['total_weeks']} weeks)

Confidence: 94%"""

ax.text(10, 50, info_text, 
        color='white', 
        fontsize=14, 
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
        verticalalignment='top',
        fontweight='bold')

ax.axis('off')
ax.set_title('Fetal Head Circumference Analysis with Gestational Age', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('sample_hc_with_gestational_age.png', dpi=150, bbox_inches='tight')
print(f"✓ Created sample_hc_with_gestational_age.png")
print(f"\nSample Calculation:")
print(f"  HC: {hc_mm} mm")
print(f"  Gestational Age: {ga_result['formatted']}")
print(f"  Total weeks: {ga_result['total_weeks']}")
