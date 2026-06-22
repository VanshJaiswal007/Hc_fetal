"""
Ellipse Fitting Utilities
Fit ellipse to binary mask and calculate circumference
"""
import cv2
import numpy as np

class EllipseFitter:
    """Wrapper class for ellipse fitting operations"""
    
    def fit(self, mask):
        """Fit ellipse to mask"""
        return fit_ellipse_to_mask(mask)
    
    def calculate_circumference(self, ellipse_params, pixel_spacing):
        """Calculate ellipse circumference"""
        return calculate_circumference(ellipse_params, pixel_spacing)
    
    def calculate_area(self, ellipse_params, pixel_spacing):
        """Calculate ellipse area"""
        return calculate_area(ellipse_params, pixel_spacing)
    
    def process_mask(self, mask, pixel_size=1.0, method='standard'):
        """
        Process a mask to extract ellipse parameters and measurements
        
        Args:
            mask: Binary mask (numpy array, values 0-255 or 0-1)
            pixel_size: Pixel spacing in mm/pixel
            method: Fitting method ('standard' or 'robust')
        
        Returns:
            Dictionary with results, success flag, and measurements
        """
        try:
            # Normalize mask to 0-1 range if needed
            if mask.max() > 1:
                mask_normalized = mask.astype(np.uint8)
            else:
                mask_normalized = (mask * 255).astype(np.uint8)
            
            # Fit ellipse
            ellipse_params = fit_ellipse_to_mask(mask_normalized)
            
            if ellipse_params is None:
                return {
                    'success': False,
                    'error_message': 'Could not fit ellipse to mask',
                    'ellipse_parameters': None,
                    'hc_measurements': None
                }
            
            # Calculate measurements
            hc_mm = calculate_circumference(ellipse_params, pixel_size)
            
            if hc_mm is None:
                return {
                    'success': False,
                    'error_message': 'Could not calculate head circumference',
                    'ellipse_parameters': None,
                    'hc_measurements': None
                }
            
            # Convert ellipse parameters to expected format
            ellipse_output = {
                'center_x': float(ellipse_params['center_x']),
                'center_y': float(ellipse_params['center_y']),
                'semi_major_axis': float(ellipse_params['major_axis'] / 2.0),
                'semi_minor_axis': float(ellipse_params['minor_axis'] / 2.0),
                'angle_degrees': float(ellipse_params['angle']),
                'major_axis_mm': float(ellipse_params['major_axis'] * pixel_size),
                'minor_axis_mm': float(ellipse_params['minor_axis'] * pixel_size)
            }
            
            # Calculate additional measurements
            # BPD (Biparietal Diameter) - typically the minor axis
            bpd_mm = ellipse_output['minor_axis_mm']
            
            # OFD (Occipitofrontal Diameter) - typically the major axis
            ofd_mm = ellipse_output['major_axis_mm']
            
            measurements = {
                'head_circumference_mm': float(hc_mm),
                'biparietal_diameter_mm': float(bpd_mm),
                'occipitofrontal_diameter_mm': float(ofd_mm),
                'area_mm2': float(calculate_area(ellipse_params, pixel_size))
            }
            
            return {
                'success': True,
                'error_message': None,
                'ellipse_parameters': ellipse_output,
                'hc_measurements': measurements
            }
            
        except Exception as e:
            return {
                'success': False,
                'error_message': f'Error processing mask: {str(e)}',
                'ellipse_parameters': None,
                'hc_measurements': None
            }

def fit_ellipse_to_mask(mask):
    """
    Fit an ellipse to a binary mask
    
    Args:
        mask: Binary mask (numpy array)
    
    Returns:
        Dictionary with ellipse parameters or None if fitting fails
    """
    try:
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Need at least 5 points to fit ellipse
        if len(largest_contour) < 5:
            return None
        
        # Fit ellipse
        ellipse = cv2.fitEllipse(largest_contour)
        
        # Extract parameters
        (center_x, center_y), (minor_axis, major_axis), angle = ellipse
        
        # Ensure major axis is the larger one
        if minor_axis > major_axis:
            minor_axis, major_axis = major_axis, minor_axis
            angle = (angle + 90) % 180
        
        params = {
            'center_x': center_x,
            'center_y': center_y,
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'angle': angle
        }
        
        return params
        
    except Exception as e:
        print(f"Error fitting ellipse: {str(e)}")
        return None

def calculate_circumference(ellipse_params, pixel_spacing):
    """
    Calculate ellipse circumference using Ramanujan's approximation
    
    Args:
        ellipse_params: Dictionary with major_axis and minor_axis (in pixels)
        pixel_spacing: Pixel spacing in mm/pixel
    
    Returns:
        Circumference in mm
    """
    try:
        # Get semi-axes in pixels
        a = ellipse_params['major_axis'] / 2.0
        b = ellipse_params['minor_axis'] / 2.0
        
        # Convert to mm
        a_mm = a * pixel_spacing
        b_mm = b * pixel_spacing
        
        # Ramanujan's approximation for ellipse circumference
        # C ≈ π * (3(a + b) - sqrt((3a + b)(a + 3b)))
        h = ((a_mm - b_mm) ** 2) / ((a_mm + b_mm) ** 2)
        circumference = np.pi * (a_mm + b_mm) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        
        return circumference
        
    except Exception as e:
        print(f"Error calculating circumference: {str(e)}")
        return None

def calculate_area(ellipse_params, pixel_spacing):
    """
    Calculate ellipse area
    
    Args:
        ellipse_params: Dictionary with major_axis and minor_axis (in pixels)
        pixel_spacing: Pixel spacing in mm/pixel
    
    Returns:
        Area in mm²
    """
    try:
        # Get semi-axes in pixels
        a = ellipse_params['major_axis'] / 2.0
        b = ellipse_params['minor_axis'] / 2.0
        
        # Convert to mm
        a_mm = a * pixel_spacing
        b_mm = b * pixel_spacing
        
        # Area = π * a * b
        area = np.pi * a_mm * b_mm
        
        return area
        
    except Exception as e:
        print(f"Error calculating area: {str(e)}")
        return None
