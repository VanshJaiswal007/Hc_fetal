"""
CSV Results Exporter
Exports detailed ellipse parameters and measurements to CSV
"""
import csv
import os
from datetime import datetime


class ResultsCSVExporter:
    """Export results to CSV with ellipse parameters"""
    
    def __init__(self, csv_path='results.csv'):
        """
        Initialize CSV exporter
        
        Args:
            csv_path: Path to the results CSV file
        """
        self.csv_path = csv_path
        self.headers = [
            'filename',
            'cx',
            'cy',
            'a',
            'b',
            'theta',
            'HC(mm)',
            'BPD(mm)',
            'OFD(mm)',
            'confidence',
            'timestamp'
        ]
        
        # Create CSV with headers if it doesn't exist
        if not os.path.exists(csv_path):
            self._create_csv()
    
    def _create_csv(self):
        """Create CSV file with headers"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
    
    def append_result(self, data):
        """
        Append a result to the CSV
        
        Args:
            data: Dictionary containing:
                - filename: Original filename
                - ellipse_params: Ellipse parameters (cx, cy, a, b, theta)
                - head_circumference_mm: HC measurement
                - measurements: Dict with BPD and OFD
                - confidence: Confidence metrics
                - timestamp: Processing timestamp
        """
        try:
            ellipse_params = data.get('ellipse_params', {})
            measurements = data.get('measurements', {})
            confidence = data.get('confidence', {})
            
            # Extract values with defaults
            row = [
                data.get('filename', 'N/A'),
                f"{ellipse_params.get('center_x', 0):.1f}" if ellipse_params else 'N/A',
                f"{ellipse_params.get('center_y', 0):.1f}" if ellipse_params else 'N/A',
                f"{ellipse_params.get('semi_major_axis', 0):.1f}" if ellipse_params else 'N/A',
                f"{ellipse_params.get('semi_minor_axis', 0):.1f}" if ellipse_params else 'N/A',
                f"{ellipse_params.get('angle_degrees', 0):.1f}" if ellipse_params else 'N/A',
                f"{data.get('head_circumference_mm', 0):.1f}" if data.get('head_circumference_mm') else 'N/A',
                f"{measurements.get('biparietal_diameter_mm', 0):.1f}" if measurements.get('biparietal_diameter_mm') else 'N/A',
                f"{measurements.get('occipitofrontal_diameter_mm', 0):.1f}" if measurements.get('occipitofrontal_diameter_mm') else 'N/A',
                f"{confidence.get('confidence_percentage', 0):.2f}" if confidence.get('confidence_percentage') is not None else 'N/A',
                data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            ]
            
            # Append to CSV
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            return True
            
        except Exception as e:
            print(f"Error appending to CSV: {e}")
            return False
    
    def get_csv_path(self):
        """Get the path to the CSV file"""
        return self.csv_path
