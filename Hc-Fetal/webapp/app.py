"""
Fetal Head Circumference Web Application
Handles DICOM/PNG/JPG images, extracts pixel spacing, performs segmentation
"""
import os
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# Try importing heavy dependencies, but don't fail if they're not available
try:
    from utils.dicom_processor import DICOMProcessor
    from utils.image_processor import ImageProcessor
    from utils.model_inference import ModelInference
    from utils.pdf_generator import FetalHCReportGenerator
    from utils.csv_exporter import ResultsCSVExporter
    DEPENDENCIES_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False
    # Create dummy classes for graceful degradation
    class DICOMProcessor: pass
    class ImageProcessor: pass
    class ModelInference: pass
    class FetalHCReportGenerator: pass
    class ResultsCSVExporter: pass

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['PROCESSED_FOLDER'] = 'data/processed'
app.config['ALLOWED_EXTENSIONS'] = {'dcm', 'dicom', 'png', 'jpg', 'jpeg'}

# Initialize processors
try:
    dicom_processor = DICOMProcessor()
    image_processor = ImageProcessor()
    model_inference = ModelInference(model_path='models/checkpoint_epoch_91.pth')
    pdf_generator = FetalHCReportGenerator()
    csv_exporter = ResultsCSVExporter(csv_path='data/results.csv')
except Exception as e:
    print(f"[WARNING] Could not initialize processors: {e}")
    dicom_processor = None
    image_processor = None
    model_inference = None
    pdf_generator = None
    csv_exporter = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/status')
def status():
    """Return application status"""
    return jsonify({
        'status': 'running',
        'dependencies_available': DEPENDENCIES_AVAILABLE,
        'message': 'HC-FETAL Web Application is running',
        'features': {
            'image_upload': True,
            'dicom_processing': DEPENDENCIES_AVAILABLE,
            'model_inference': DEPENDENCIES_AVAILABLE,
            'pdf_generation': DEPENDENCIES_AVAILABLE,
            'csv_export': DEPENDENCIES_AVAILABLE
        }
    }), 200

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create session folders
        session_folder = os.path.join(app.config['PROCESSED_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # Save original file
        original_filename = secure_filename(file.filename)
        file_ext = original_filename.rsplit('.', 1)[1].lower()
        original_path = os.path.join(session_folder, f'original_{timestamp}.{file_ext}')
        file.save(original_path)
        
        # Process based on file type
        if file_ext in ['dcm', 'dicom']:
            result = process_dicom(original_path, session_folder, timestamp)
        else:
            result = process_image(original_path, session_folder, timestamp)
        
        result['session_id'] = session_id
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_dicom(dicom_path, session_folder, timestamp):
    """Process DICOM file"""
    # 1. Extract metadata and pixel spacing
    metadata = dicom_processor.extract_metadata(dicom_path)
    
    # 2. Convert to PNG with windowing
    png_path = os.path.join(session_folder, f'converted_{timestamp}.png')
    dicom_processor.convert_to_png(dicom_path, png_path, apply_windowing=True)
    
    # 3. Export metadata to CSV
    csv_path = os.path.join(session_folder, f'metadata_{timestamp}.csv')
    dicom_processor.export_metadata_csv(metadata, csv_path)
    
    # 4. Run model inference
    prediction_result = model_inference.predict(png_path, metadata['pixel_spacing'])
    
    # 5. Save prediction visualization
    viz_path = os.path.join(session_folder, f'prediction_{timestamp}.png')
    model_inference.save_visualization(png_path, prediction_result, viz_path)
    
    # 6. Save result data for later PDF generation
    import json
    import numpy as np
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    result_data = {
        'original_file': os.path.basename(dicom_path),
        'file_type': 'dicom',
        'metadata': convert_to_serializable(metadata),
        'pixel_spacing': convert_to_serializable(metadata['pixel_spacing']),
        'head_circumference_mm': float(prediction_result['head_circumference_mm']) if prediction_result['head_circumference_mm'] else None,
        'measurements': convert_to_serializable(prediction_result.get('hc_measurements', {})),
        'confidence': convert_to_serializable(prediction_result.get('confidence', {})),
        'ellipse_params': convert_to_serializable(prediction_result.get('ellipse_params', {})),
        'timestamp': timestamp
    }
    result_json_path = os.path.join(session_folder, f'result_{timestamp}.json')
    with open(result_json_path, 'w') as f:
        json.dump(result_data, f)
    
    return {
        'status': 'success',
        'file_type': 'dicom',
        'original_file': os.path.basename(dicom_path),
        'converted_image': os.path.basename(png_path),
        'metadata_csv': os.path.basename(csv_path),
        'prediction_image': os.path.basename(viz_path),
        'pixel_spacing': metadata['pixel_spacing'],
        'head_circumference_mm': prediction_result['head_circumference_mm'],
        'confidence': prediction_result.get('confidence', {}),
        'metadata': metadata
    }

def process_image(image_path, session_folder, timestamp):
    """Process PNG/JPG file"""
    # 1. Extract pixel spacing from image metadata
    pixel_spacing = image_processor.extract_pixel_spacing(image_path)
    
    if pixel_spacing is None:
        # Prompt user for pixel spacing
        return {
            'status': 'needs_pixel_spacing',
            'message': 'Could not extract pixel spacing from image. Please provide it manually.',
            'original_file': os.path.basename(image_path)
        }
    
    # 2. Create metadata
    metadata = image_processor.create_metadata(image_path, pixel_spacing)
    
    # 3. Export metadata to CSV
    csv_path = os.path.join(session_folder, f'metadata_{timestamp}.csv')
    image_processor.export_metadata_csv(metadata, csv_path)
    
    # 4. Run model inference
    prediction_result = model_inference.predict(image_path, pixel_spacing)
    
    # 5. Save prediction visualization
    viz_path = os.path.join(session_folder, f'prediction_{timestamp}.png')
    model_inference.save_visualization(image_path, prediction_result, viz_path)
    
    # 6. Save result data for later PDF generation
    import json
    import numpy as np
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    result_data = {
        'original_file': os.path.basename(image_path),
        'file_type': 'image',
        'metadata': convert_to_serializable(metadata),
        'pixel_spacing': float(pixel_spacing) if not isinstance(pixel_spacing, (list, tuple)) else [float(x) for x in pixel_spacing],
        'head_circumference_mm': float(prediction_result['head_circumference_mm']) if prediction_result['head_circumference_mm'] else None,
        'measurements': convert_to_serializable(prediction_result.get('hc_measurements', {})),
        'confidence': convert_to_serializable(prediction_result.get('confidence', {})),
        'ellipse_params': convert_to_serializable(prediction_result.get('ellipse_params', {})),
        'timestamp': timestamp
    }
    result_json_path = os.path.join(session_folder, f'result_{timestamp}.json')
    with open(result_json_path, 'w') as f:
        json.dump(result_data, f)
    
    return {
        'status': 'success',
        'file_type': 'image',
        'original_file': os.path.basename(image_path),
        'metadata_csv': os.path.basename(csv_path),
        'prediction_image': os.path.basename(viz_path),
        'pixel_spacing': pixel_spacing,
        'head_circumference_mm': prediction_result['head_circumference_mm'],
        'confidence': prediction_result.get('confidence', {}),
        'metadata': metadata
    }

@app.route('/api/generate_pdf', methods=['POST'])
def generate_pdf_with_patient_id():
    """Generate PDF report with patient ID"""
    data = request.json
    session_id = data.get('session_id')
    patient_id = data.get('patient_id', '')
    
    if not session_id:
        return jsonify({'error': 'Missing session_id'}), 400
    
    try:
        session_folder = os.path.join(app.config['PROCESSED_FOLDER'], session_id)
        
        # Find the latest prediction and metadata
        files = os.listdir(session_folder)
        prediction_files = [f for f in files if f.startswith('prediction_')]
        
        if not prediction_files:
            return jsonify({'error': 'No prediction found'}), 404
        
        # Get the latest prediction
        prediction_files.sort(reverse=True)
        prediction_file = prediction_files[0]
        timestamp = prediction_file.replace('prediction_', '').replace('.png', '')
        
        # Load stored result data (we'll need to save this during processing)
        import json
        result_json_path = os.path.join(session_folder, f'result_{timestamp}.json')
        
        if os.path.exists(result_json_path):
            with open(result_json_path, 'r') as f:
                result_data = json.load(f)
        else:
            return jsonify({'error': 'Result data not found'}), 404
        
        # Generate PDF with patient info
        pdf_path = os.path.join(session_folder, f'report_{patient_id}_{timestamp}.pdf')
        
        pdf_data = result_data.copy()
        pdf_data['patient_info'] = {
            'patient_id': patient_id,
            'exam_date': datetime.now().strftime('%d %b %Y')
        }
        pdf_data['prediction_image_path'] = os.path.join(session_folder, prediction_file)
        
        pdf_generator.generate_report(pdf_path, pdf_data)
        
        # Export to CSV
        csv_data = {
            'filename': result_data.get('original_file', 'N/A'),
            'ellipse_params': result_data.get('ellipse_params'),
            'head_circumference_mm': result_data.get('head_circumference_mm'),
            'measurements': result_data.get('measurements', {}),
            'confidence': result_data.get('confidence', {}),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        csv_exporter.append_result(csv_data)
        
        return jsonify({
            'status': 'success',
            'pdf_report': os.path.basename(pdf_path)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/set_pixel_spacing', methods=['POST'])
def set_pixel_spacing():
    """Manually set pixel spacing for images without metadata"""
    data = request.json
    session_id = data.get('session_id')
    pixel_spacing = data.get('pixel_spacing')
    
    if not session_id or not pixel_spacing:
        return jsonify({'error': 'Missing session_id or pixel_spacing'}), 400
    
    try:
        session_folder = os.path.join(app.config['PROCESSED_FOLDER'], session_id)
        # Find original image
        files = os.listdir(session_folder)
        original_file = [f for f in files if f.startswith('original_')][0]
        image_path = os.path.join(session_folder, original_file)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create metadata with manual pixel spacing
        metadata = image_processor.create_metadata(image_path, pixel_spacing)
        
        # Export metadata to CSV
        csv_path = os.path.join(session_folder, f'metadata_{timestamp}.csv')
        image_processor.export_metadata_csv(metadata, csv_path)
        
        # Run model inference
        prediction_result = model_inference.predict(image_path, pixel_spacing)
        
        # Save prediction visualization
        viz_path = os.path.join(session_folder, f'prediction_{timestamp}.png')
        model_inference.save_visualization(image_path, prediction_result, viz_path)
        
        # Save result data for later PDF generation
        import json
        import numpy as np
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        result_data = {
            'original_file': os.path.basename(image_path),
            'file_type': 'image',
            'metadata': convert_to_serializable(metadata),
            'pixel_spacing': float(pixel_spacing) if not isinstance(pixel_spacing, (list, tuple)) else [float(x) for x in pixel_spacing],
            'head_circumference_mm': float(prediction_result['head_circumference_mm']) if prediction_result['head_circumference_mm'] else None,
            'measurements': convert_to_serializable(prediction_result.get('hc_measurements', {})),
            'confidence': convert_to_serializable(prediction_result.get('confidence', {})),
            'ellipse_params': convert_to_serializable(prediction_result.get('ellipse_params', {})),
            'timestamp': timestamp
        }
        result_json_path = os.path.join(session_folder, f'result_{timestamp}.json')
        with open(result_json_path, 'w') as f:
            json.dump(result_data, f)
        
        return jsonify({
            'status': 'success',
            'metadata_csv': os.path.basename(csv_path),
            'prediction_image': os.path.basename(viz_path),
            'pixel_spacing': pixel_spacing,
            'head_circumference_mm': prediction_result['head_circumference_mm'],
            'confidence': prediction_result.get('confidence', {})
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<session_id>/<filename>')
def download_file(session_id, filename):
    """Download processed files"""
    try:
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], session_id, filename)
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/download_results_csv')
def download_results_csv():
    """Download the results CSV file"""
    try:
        csv_path = csv_exporter.get_csv_path()
        if os.path.exists(csv_path):
            return send_file(csv_path, as_attachment=True, download_name='results.csv')
        else:
            return jsonify({'error': 'Results CSV not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary folders
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
