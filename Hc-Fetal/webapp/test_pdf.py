"""
Quick test for PDF generation
"""
from utils.pdf_generator import FetalHCReportGenerator
from datetime import datetime

# Create test data
test_data = {
    'original_file': 'test_ultrasound.png',
    'file_type': 'image',
    'metadata': {
        'modality': 'US',
        'study_date': '2024-01-15',
        'institution_name': 'Test Hospital'
    },
    'pixel_spacing': [0.0691, 0.0691],
    'head_circumference_mm': 285.43,
    'measurements': {
        'head_circumference_mm': 285.43,
        'occipitofrontal_diameter_mm': 95.2,
        'biparietal_diameter_mm': 87.5,
        'area_mm2': 6234.5
    },
    'prediction_image_path': None,  # No image for this test
    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
}

# Generate PDF
generator = FetalHCReportGenerator()
output_path = 'test_report.pdf'

try:
    generator.generate_report(output_path, test_data)
    print(f"✓ PDF report generated successfully: {output_path}")
except Exception as e:
    print(f"✗ Error generating PDF: {e}")
    import traceback
    traceback.print_exc()
