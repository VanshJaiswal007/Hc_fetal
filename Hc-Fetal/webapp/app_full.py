"""
Fetal Head Circumference Web Application - FULL VERSION (Python 3.13 Compatible)
Uses lazy loading to avoid problematic imports on startup
"""
import os
import sys
from pathlib import Path
import json
import numpy as np

# Python 3.13 Compatibility: Set environment variables early
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import traceback

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Initialize Flask app
app = Flask(__name__)
app.json_encoder = NumpyEncoder  # Set custom encoder for older Flask versions
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['PROCESSED_FOLDER'] = 'data/processed'
app.config['ALLOWED_EXTENSIONS'] = {'dcm', 'dicom', 'png', 'jpg', 'jpeg'}

# Global state
DEPENDENCIES_STATE = {
    'loaded': False,
    'errors': [],
    'processors': {
        'dicom': None,
        'image': None,
        'model': None,
        'pdf': None,
        'csv': None
    }
}

def lazy_load_dependencies():
    """Lazy load dependencies only when needed"""
    print("🔧 Loading ML dependencies (this may take a moment)...")
    
    try:
        print("  • Importing DICOM processor...")
        from utils.dicom_processor import DICOMProcessor
        DEPENDENCIES_STATE['processors']['dicom'] = DICOMProcessor()
        print("    ✓ DICOM processor ready")
    except Exception as e:
        print(f"    ✗ DICOM processor: {e}")
        DEPENDENCIES_STATE['errors'].append(f"DICOM: {str(e)}")

    try:
        print("  • Importing image processor...")
        from utils.image_processor import ImageProcessor
        DEPENDENCIES_STATE['processors']['image'] = ImageProcessor()
        print("    ✓ Image processor ready")
    except Exception as e:
        print(f"    ✗ Image processor: {e}")
        DEPENDENCIES_STATE['errors'].append(f"ImageProcessor: {str(e)}")

    try:
        print("  • Importing model inference...")
        from utils.model_inference import ModelInference
        model_path = os.path.join(Path(__file__).parent.parent, 'models', 'checkpoint_epoch_91.pth')
        DEPENDENCIES_STATE['processors']['model'] = ModelInference(model_path=model_path if os.path.exists(model_path) else None)
        print("    ✓ Model inference ready")
    except Exception as e:
        print(f"    ✗ Model inference: {e}")
        DEPENDENCIES_STATE['errors'].append(f"ModelInference: {str(e)}")

    try:
        print("  • Importing PDF generator...")
        from utils.pdf_generator import FetalHCReportGenerator
        DEPENDENCIES_STATE['processors']['pdf'] = FetalHCReportGenerator()
        print("    ✓ PDF generator ready")
    except Exception as e:
        print(f"    ✗ PDF generator: {e}")
        DEPENDENCIES_STATE['errors'].append(f"PDFGenerator: {str(e)}")

    try:
        print("  • Importing CSV exporter...")
        from utils.csv_exporter import ResultsCSVExporter
        DEPENDENCIES_STATE['processors']['csv'] = ResultsCSVExporter(csv_path='data/results.csv')
        print("    ✓ CSV exporter ready")
    except Exception as e:
        print(f"    ✗ CSV exporter: {e}")
        DEPENDENCIES_STATE['errors'].append(f"CSVExporter: {str(e)}")
    
    DEPENDENCIES_STATE['loaded'] = True
    
    if DEPENDENCIES_STATE['errors']:
        print(f"\n⚠️  {len(DEPENDENCIES_STATE['errors'])} dependency issues:")
        for err in DEPENDENCIES_STATE['errors']:
            print(f"  • {err}")
    else:
        print("\n✅ All ML dependencies loaded successfully!")
    
    print()



def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/api/status')
def api_status():
    """Return application status"""
    return jsonify({
        'status': 'running',
        'version': 'full',
        'message': 'HC-FETAL Web Application (Full Version) is running',
        'python_version': sys.version.split()[0],
        'dependencies_loaded': DEPENDENCIES_STATE['loaded'],
        'features': {
            'image_upload': True,
            'dicom_processing': DEPENDENCIES_STATE['processors']['dicom'] is not None,
            'model_inference': DEPENDENCIES_STATE['processors']['model'] is not None,
            'pdf_generation': DEPENDENCIES_STATE['processors']['pdf'] is not None,
            'csv_export': DEPENDENCIES_STATE['processors']['csv'] is not None
        },
        'errors': DEPENDENCIES_STATE['errors'] if DEPENDENCIES_STATE['errors'] else []
    }), 200

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/')
def index():
    """Serve main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        # Fallback HTML if template not found
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fetal Head Circumference</title>
            <style>
                body { font-family: Arial; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .status { padding: 20px; margin: 20px 0; border-radius: 5px; }
                .success { background: #d4edda; color: #155724; }
                .warning { background: #fff3cd; color: #856404; }
                h1 { color: #333; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎉 Fetal Head Circumference Analysis</h1>
                <p>Upload DICOM, PNG, or JPG ultrasound images for automatic HC measurement</p>
                
                <div class="status success">
                    <strong>✅ Application Status:</strong> Running (Full Version)
                </div>
                
                <div class="status info">
                    <strong>💡 Status:</strong> ML dependencies are loaded on first image upload
                </div>
                
                <p><a href="/api/status">Check API Status</a></p>
                <p><a href="/api/health">Health Check</a></p>
            </div>
        </body>
        </html>
        '''

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type',
            'allowed_types': list(app.config['ALLOWED_EXTENSIONS'])
        }), 400
    
    try:
        # Lazy load dependencies on first upload
        if not DEPENDENCIES_STATE['loaded']:
            lazy_load_dependencies()
        
        # Generate session
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_folder = os.path.join(app.config['PROCESSED_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # Save file
        original_filename = secure_filename(file.filename)
        file_ext = original_filename.rsplit('.', 1)[1].lower()
        original_path = os.path.join(session_folder, f'original_{timestamp}.{file_ext}')
        file.save(original_path)
        
        response = {
            'success': True,
            'session_id': session_id,
            'file_name': original_filename,
            'file_path': original_path,
            'timestamp': timestamp
        }
        
        # Get pixel spacing from DICOM
        pixel_spacing = 1.0
        if file_ext in ['dcm', 'dicom'] and DEPENDENCIES_STATE['processors']['dicom']:
            try:
                dicom_data = DEPENDENCIES_STATE['processors']['dicom'].read_dicom(original_path)
                if dicom_data and dicom_data.get('pixel_spacing'):
                    pixel_spacing = dicom_data['pixel_spacing']
                    response['pixel_spacing'] = pixel_spacing
            except Exception as e:
                print(f"⚠ DICOM read warning: {e}")
        
        # Process image
        if DEPENDENCIES_STATE['processors']['image']:
            try:
                # Validate image first
                is_valid = DEPENDENCIES_STATE['processors']['image'].validate_image(original_path)
                if is_valid:
                    # Extract pixel spacing from metadata
                    extracted_spacing = DEPENDENCIES_STATE['processors']['image'].extract_pixel_spacing(original_path)
                    if extracted_spacing:
                        pixel_spacing = extracted_spacing
                        response['pixel_spacing_extracted'] = True
            except Exception as e:
                print(f"⚠ Image processing warning: {e}")
        
        # Run inference
        if DEPENDENCIES_STATE['processors']['model']:
            try:
                results = DEPENDENCIES_STATE['processors']['model'].predict(original_path, pixel_spacing=pixel_spacing)
                response['results'] = results
                
                # Save results to JSON
                results_path = os.path.join(session_folder, 'results.json')
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Generate PDF
                if DEPENDENCIES_STATE['processors']['pdf']:
                    try:
                        pdf_path = os.path.join(session_folder, 'report.pdf')
                        DEPENDENCIES_STATE['processors']['pdf'].generate(
                            output_path=pdf_path,
                            original_image=original_path,
                            results=results
                        )
                        response['pdf_path'] = pdf_path
                        response['pdf_available'] = True
                    except Exception as e:
                        print(f"⚠ PDF generation warning: {e}")
                
                # Export to CSV
                if DEPENDENCIES_STATE['processors']['csv']:
                    try:
                        DEPENDENCIES_STATE['processors']['csv'].export(session_id, results, original_path)
                    except Exception as e:
                        print(f"⚠ CSV export warning: {e}")
                
            except Exception as e:
                print(f"⚠ Model inference warning: {e}")
                response['inference_available'] = False
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ Upload error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/download/<file_type>/<session_id>')
def download_file(file_type, session_id):
    """Download generated files (PDF, CSV)"""
    try:
        session_folder = os.path.join(app.config['PROCESSED_FOLDER'], session_id)
        
        if file_type == 'pdf':
            file_path = os.path.join(session_folder, 'report.pdf')
            if os.path.exists(file_path):
                return send_file(file_path, mimetype='application/pdf')
        
        elif file_type == 'csv':
            file_path = os.path.join(session_folder, 'results.csv')
            if os.path.exists(file_path):
                return send_file(file_path, mimetype='text/csv')
        
        return jsonify({'error': 'File not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/set_pixel_spacing', methods=['POST'])
def set_pixel_spacing():
    """Manually set pixel spacing for a session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        pixel_spacing = float(data.get('pixel_spacing', 1.0))
        
        # Store in session config
        session_config_path = os.path.join(
            app.config['PROCESSED_FOLDER'],
            session_id,
            'config.json'
        )
        
        os.makedirs(os.path.dirname(session_config_path), exist_ok=True)
        
        import json
        with open(session_config_path, 'w') as f:
            json.dump({'pixel_spacing': pixel_spacing}, f)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'pixel_spacing': pixel_spacing
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/generate_pdf', methods=['POST'])
def generate_pdf():
    """Generate PDF for existing session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not DEPENDENCIES_STATE['processors']['pdf']:
            return jsonify({'error': 'PDF generation not available'}), 503
        
        session_folder = os.path.join(app.config['PROCESSED_FOLDER'], session_id)
        
        # Read results from this session
        results_file = os.path.join(session_folder, 'results.json')
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            pdf_path = os.path.join(session_folder, 'report.pdf')
            DEPENDENCIES_STATE['processors']['pdf'].generate(
                output_path=pdf_path,
                results=results
            )
            
            return jsonify({
                'success': True,
                'pdf_path': pdf_path
            }), 200
        
        return jsonify({'error': 'Results not found for session'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create required folders
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    
    print("\n" + "="*70)
    print("🎉 HC-FETAL Web Application (FULL VERSION)")
    print("="*70)
    print(f"\n📱 Web Interface: http://localhost:5000")
    print(f"📊 API Status: http://localhost:5000/api/status")
    print(f"💚 Health Check: http://localhost:5000/api/health")
    print(f"\n🔧 Python Version: {sys.version.split()[0]}")
    print(f"📦 ML Dependencies: Loaded on first image upload (lazy loading)")
    print("\n" + "="*70)
    print("\n💡 Tip: Dependencies will be loaded when you upload your first image.\n")
    
    # Run Flask app
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
