"""
DICOM Processing Utilities
Handles DICOM reading, metadata extraction, windowing, and PNG conversion
"""
import pydicom
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os

class DICOMProcessor:
    def __init__(self):
        self.default_window_center = 127
        self.default_window_width = 255
    
    def extract_metadata(self, dicom_path):
        """Extract all relevant metadata from DICOM file"""
        try:
            dcm = pydicom.dcmread(dicom_path)
            
            # Extract pixel spacing (priority order)
            pixel_spacing = self._get_pixel_spacing(dcm)
            
            metadata = {
                'filename': os.path.basename(dicom_path),
                'patient_id': str(getattr(dcm, 'PatientID', 'Unknown')),
                'patient_name': str(getattr(dcm, 'PatientName', 'Unknown')),
                'study_date': str(getattr(dcm, 'StudyDate', 'Unknown')),
                'study_time': str(getattr(dcm, 'StudyTime', 'Unknown')),
                'modality': str(getattr(dcm, 'Modality', 'Unknown')),
                'manufacturer': str(getattr(dcm, 'Manufacturer', 'Unknown')),
                'institution': str(getattr(dcm, 'InstitutionName', 'Unknown')),
                'rows': int(dcm.Rows),
                'columns': int(dcm.Columns),
                'pixel_spacing': pixel_spacing,
                'bits_allocated': int(getattr(dcm, 'BitsAllocated', 0)),
                'bits_stored': int(getattr(dcm, 'BitsStored', 0)),
                'photometric_interpretation': str(getattr(dcm, 'PhotometricInterpretation', 'Unknown')),
                'window_center': self._get_window_center(dcm),
                'window_width': self._get_window_width(dcm),
                'processed_timestamp': datetime.now().isoformat()
            }
            
            return metadata
            
        except Exception as e:
            raise Exception(f"Error extracting DICOM metadata: {str(e)}")
    
    def _get_pixel_spacing(self, dcm):
        """Get pixel spacing from DICOM with fallback options"""
        # Try PixelSpacing first (most common)
        if hasattr(dcm, 'PixelSpacing'):
            spacing = dcm.PixelSpacing
            return float(spacing[0])  # Assuming square pixels
        
        # Try ImagerPixelSpacing
        if hasattr(dcm, 'ImagerPixelSpacing'):
            spacing = dcm.ImagerPixelSpacing
            return float(spacing[0])
        
        # Try SequenceOfUltrasoundRegions (for ultrasound)
        if hasattr(dcm, 'SequenceOfUltrasoundRegions'):
            regions = dcm.SequenceOfUltrasoundRegions
            if len(regions) > 0 and hasattr(regions[0], 'PhysicalDeltaX'):
                return float(regions[0].PhysicalDeltaX) * 10  # Convert cm to mm
        
        # No pixel spacing found
        return None
    
    def _get_window_center(self, dcm):
        """Get window center for display"""
        if hasattr(dcm, 'WindowCenter'):
            wc = dcm.WindowCenter
            return float(wc[0]) if isinstance(wc, (list, tuple)) else float(wc)
        return self.default_window_center
    
    def _get_window_width(self, dcm):
        """Get window width for display"""
        if hasattr(dcm, 'WindowWidth'):
            ww = dcm.WindowWidth
            return float(ww[0]) if isinstance(ww, (list, tuple)) else float(ww)
        return self.default_window_width
    
    def convert_to_png(self, dicom_path, output_path, apply_windowing=True):
        """Convert DICOM to PNG with optional windowing"""
        try:
            dcm = pydicom.dcmread(dicom_path)
            pixel_array = dcm.pixel_array.astype(float)
            
            if apply_windowing:
                # Apply window/level
                window_center = self._get_window_center(dcm)
                window_width = self._get_window_width(dcm)
                
                img_min = window_center - window_width / 2
                img_max = window_center + window_width / 2
                
                pixel_array = np.clip(pixel_array, img_min, img_max)
                pixel_array = ((pixel_array - img_min) / (img_max - img_min) * 255.0)
            else:
                # Simple normalization
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255.0)
            
            # Convert to uint8
            pixel_array = pixel_array.astype(np.uint8)
            
            # Handle photometric interpretation
            if hasattr(dcm, 'PhotometricInterpretation'):
                if dcm.PhotometricInterpretation == "MONOCHROME1":
                    pixel_array = 255 - pixel_array  # Invert
            
            # Save as PNG
            img = Image.fromarray(pixel_array)
            img.save(output_path)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error converting DICOM to PNG: {str(e)}")
    
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
    
    def apply_windowing(self, pixel_array, window_center, window_width):
        """Apply window/level transformation"""
        img_min = window_center - window_width / 2
        img_max = window_center + window_width / 2
        
        windowed = np.clip(pixel_array, img_min, img_max)
        windowed = ((windowed - img_min) / (img_max - img_min) * 255.0)
        
        return windowed.astype(np.uint8)
