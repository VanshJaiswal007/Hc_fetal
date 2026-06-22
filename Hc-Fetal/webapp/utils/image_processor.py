"""
Image Processing Utilities
Handles PNG/JPG images, extracts pixel spacing from metadata
"""
from PIL import Image
import pandas as pd
from datetime import datetime
import os

class ImageProcessor:
    def extract_pixel_spacing(self, image_path):
        """Extract pixel spacing from PNG/JPG metadata"""
        try:
            img = Image.open(image_path)
            
            # Method 1: Check for 'aspect' metadata (like HC18 dataset)
            if 'aspect' in img.info:
                aspect_value = img.info['aspect'][0]
                pixel_spacing = 1000.0 / aspect_value  # Convert to mm/pixel
                return pixel_spacing
            
            # Method 2: Check for DPI metadata
            elif 'dpi' in img.info:
                dpi = img.info['dpi'][0]
                pixel_spacing = 25.4 / dpi  # 25.4 mm per inch
                return pixel_spacing
            
            # Method 3: Check for resolution unit
            elif hasattr(img, 'info') and 'resolution' in img.info:
                # Some images store resolution differently
                resolution = img.info['resolution']
                if isinstance(resolution, tuple):
                    pixel_spacing = 25.4 / resolution[0]
                    return pixel_spacing
            
            # No pixel spacing found
            return None
            
        except Exception as e:
            print(f"Error extracting pixel spacing: {str(e)}")
            return None
    
    def create_metadata(self, image_path, pixel_spacing):
        """Create metadata dictionary for image"""
        try:
            img = Image.open(image_path)
            
            metadata = {
                'filename': os.path.basename(image_path),
                'format': img.format,
                'mode': img.mode,
                'width': img.size[0],
                'height': img.size[1],
                'pixel_spacing': pixel_spacing,
                'file_size_bytes': os.path.getsize(image_path),
                'processed_timestamp': datetime.now().isoformat()
            }
            
            # Add any additional info from image
            if hasattr(img, 'info'):
                for key, value in img.info.items():
                    if key not in metadata and isinstance(value, (str, int, float)):
                        metadata[f'image_{key}'] = value
            
            return metadata
            
        except Exception as e:
            raise Exception(f"Error creating metadata: {str(e)}")
    
    def export_metadata_csv(self, metadata, csv_path):
        """Export metadata to CSV file"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([metadata])
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            return csv_path
            
        except Exception as e:
            raise Exception(f"Error exporting metadata to CSV: {str(e)}")
    
    def validate_image(self, image_path):
        """Validate if image is suitable for processing"""
        try:
            img = Image.open(image_path)
            
            # Check if grayscale or can be converted
            if img.mode not in ['L', 'RGB', 'RGBA']:
                return False, f"Unsupported image mode: {img.mode}"
            
            # Check minimum size
            if img.size[0] < 100 or img.size[1] < 100:
                return False, "Image too small (minimum 100x100 pixels)"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
