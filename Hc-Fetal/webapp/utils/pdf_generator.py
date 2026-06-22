"""
PDF Report Generator for Fetal Head Circumference Analysis
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from datetime import datetime
import os


class FetalHCReportGenerator:
    """Generate professional PDF reports for fetal head circumference analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Info style
        self.styles.add(ParagraphStyle(
            name='InfoText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            fontName='Helvetica'
        ))
    
    def generate_report(self, output_path, data):
        """
        Generate PDF report
        
        Args:
            output_path: Path to save the PDF
            data: Dictionary containing:
                - patient_info: dict with patient details (optional)
                - metadata: dict with image metadata
                - measurements: dict with HC measurements
                - prediction_image_path: path to prediction visualization
                - original_file: original filename
                - timestamp: analysis timestamp
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        story = []
        
        # Header
        story.extend(self._create_header(data))
        
        # Patient Information (if available)
        if data.get('patient_info'):
            story.extend(self._create_patient_info(data['patient_info']))
        
        # Image Information
        story.extend(self._create_image_info(data))
        
        # Measurements Section
        story.extend(self._create_measurements_section(data))
        
        # Prediction Image
        story.extend(self._create_image_section(data))
        
        # Footer
        story.extend(self._create_footer(data))
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def _create_header(self, data):
        """Create report header"""
        elements = []
        
        # Separator line
        separator = "—" * 80
        elements.append(Paragraph(separator, self.styles['InfoText']))
        
        # Title
        title = Paragraph("Fetal Biometry AI Report", self.styles['CustomTitle'])
        elements.append(title)
        
        # Separator line
        elements.append(Paragraph(separator, self.styles['InfoText']))
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_patient_info(self, patient_info):
        """Create patient information section"""
        elements = []
        
        # Simple text format
        info_lines = []
        
        if patient_info.get('patient_id'):
            info_lines.append(f"<b>Patient ID:</b> {patient_info['patient_id']}")
        
        if patient_info.get('exam_date'):
            info_lines.append(f"<b>Date:</b> {patient_info['exam_date']}")
        
        for line in info_lines:
            elements.append(Paragraph(line, self.styles['InfoText']))
        
        if info_lines:
            elements.append(Spacer(1, 0.15*inch))
        
        return elements
    
    def _create_image_info(self, data):
        """Create image information section - removed for simplified format"""
        return []
    
    def _create_measurements_section(self, data):
        """Create measurements section - simplified format"""
        elements = []
        
        measurements = data.get('measurements', {})
        hc_mm = data.get('head_circumference_mm') or measurements.get('head_circumference_mm')
        
        if hc_mm is None:
            elements.append(Paragraph("No measurements available", self.styles['InfoText']))
            elements.append(Spacer(1, 0.2*inch))
            return elements
        
        # Simple text format for measurements
        measurement_lines = []
        measurement_lines.append(f"<b>Head Circumference (HC):</b> {hc_mm:.1f} mm")
        
        if measurements.get('biparietal_diameter_mm'):
            bpd = measurements['biparietal_diameter_mm']
            measurement_lines.append(f"<b>Biparietal Diameter (BPD):</b> {bpd:.1f} mm")
        
        # Add confidence
        confidence = data.get('confidence', {})
        if confidence.get('confidence_percentage') is not None:
            conf_percent = confidence['confidence_percentage']
            measurement_lines.append(f"<b>Confidence:</b> {conf_percent:.0f}%")
        
        for line in measurement_lines:
            elements.append(Paragraph(line, self.styles['InfoText']))
        
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_image_section(self, data):
        """Create image section with prediction visualization"""
        elements = []
        
        prediction_image_path = data.get('prediction_image_path')
        
        if prediction_image_path and os.path.exists(prediction_image_path):
            # Add image with appropriate sizing
            img = Image(prediction_image_path)
            
            # Scale image to fit page width (max 6 inches)
            max_width = 6 * inch
            max_height = 4.5 * inch
            
            aspect = img.imageHeight / float(img.imageWidth)
            
            if img.imageWidth > max_width:
                img.drawWidth = max_width
                img.drawHeight = max_width * aspect
            
            if img.drawHeight > max_height:
                img.drawHeight = max_height
                img.drawWidth = max_height / aspect
            
            elements.append(img)
            elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def _create_footer(self, data):
        """Create report footer"""
        elements = []
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Separator line
        separator = "—" * 80
        elements.append(Paragraph(separator, self.styles['InfoText']))
        
        # Model info
        model_text = "<b>Model:</b> UNET_ResNet34"
        elements.append(Paragraph(model_text, self.styles['InfoText']))
        
        return elements
