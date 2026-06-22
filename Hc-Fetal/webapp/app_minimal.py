"""
HC-FETAL Minimal Web Application
Lightweight version without heavy ML dependencies
"""
import os
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['PROCESSED_FOLDER'] = 'data/processed'
app.config['ALLOWED_EXTENSIONS'] = {'dcm', 'dicom', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Main page"""
    try:
        return render_template('index.html')
    except:
        # Fallback if template not found
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>HC-FETAL - Fetal Head Circumference Analyzer</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 50px auto; padding: 20px; }
                .container { background: #f5f5f5; padding: 30px; border-radius: 10px; }
                h1 { color: #2c3e50; }
                .status { background: #d4edda; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .info { background: #cce5ff; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .upload-area { 
                    border: 2px dashed #2c3e50; 
                    padding: 40px; 
                    text-align: center; 
                    border-radius: 5px;
                    margin: 20px 0;
                    cursor: pointer;
                }
                button { 
                    background: #2c3e50; 
                    color: white; 
                    padding: 10px 20px; 
                    border: none; 
                    border-radius: 5px; 
                    cursor: pointer;
                    font-size: 16px;
                }
                button:hover { background: #34495e; }
                .message { margin-top: 20px; padding: 15px; border-radius: 5px; }
                .success { background: #d4edda; color: #155724; }
                .error { background: #f8d7da; color: #721c24; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔬 HC-FETAL Application</h1>
                <h2>Fetal Head Circumference Analyzer</h2>
                
                <div class="status">
                    <strong>✅ Application Status:</strong> Running
                </div>
                
                <div class="info">
                    <h3>Application Information</h3>
                    <p>This is a lightweight version of HC-FETAL without full ML capabilities.</p>
                    <p><strong>Status:</strong> App is running successfully!</p>
                </div>
                
                <div class="upload-area">
                    <h3>Ready for Image Upload</h3>
                    <p>This interface is prepared for image processing.</p>
                    <p>Full features require Python dependencies to be installed.</p>
                </div>
                
                <div id="message"></div>
            </div>
            
            <script>
                // Check application status
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        console.log('App Status:', data);
                        const msg = document.getElementById('message');
                        if (data.dependencies_available) {
                            msg.innerHTML = '<div class="message success"><strong>✅ Full Features Available</strong><br>Model inference and PDF generation are ready.</div>';
                        } else {
                            msg.innerHTML = '<div class="message error"><strong>⚠ Dependencies Missing</strong><br>ML features are not available. Please install: pip install -r requirements.txt</div>';
                        }
                    })
                    .catch(error => console.log('Status check failed:', error));
            </script>
        </body>
        </html>
        """, 200

@app.route('/api/status')
def status():
    """Return application status"""
    return jsonify({
        'status': 'running',
        'app_name': 'HC-FETAL',
        'version': '1.0',
        'message': 'Application is running successfully',
        'note': 'Full features require Python dependencies installation'
    }), 200

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    return jsonify({
        'error': 'Full dependencies required for image processing',
        'message': 'Please install requirements: pip install -r requirements.txt'
    }), 503

if __name__ == '__main__':
    # Create necessary folders
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    
    print("\n" + "="*60)
    print("🎉 HC-FETAL Web Application Started")
    print("="*60)
    print("\n📱 Web Interface: http://localhost:5000")
    print("✅ API Status: http://localhost:5000/api/status")
    print("💚 Health Check: http://localhost:5000/api/health")
    print("\n📋 Note: This is a minimal version.")
    print("   For full features, install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
