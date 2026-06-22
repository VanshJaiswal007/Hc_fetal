"""
Startup script for the web application
Checks dependencies and starts the server
"""
import sys
import os

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    
    missing = []
    
    try:
        import flask
        print("[OK] Flask")
    except ImportError:
        missing.append("flask")
        print("[FAIL] Flask")
    
    try:
        import pydicom
        print("[OK] pydicom")
    except ImportError:
        missing.append("pydicom")
        print("[FAIL] pydicom")
    
    try:
        from PIL import Image
        print("[OK] Pillow")
    except ImportError:
        missing.append("pillow")
        print("[FAIL] Pillow")
    
    try:
        import numpy
        print("[OK] NumPy")
    except ImportError:
        missing.append("numpy")
        print("[FAIL] NumPy")
    
    try:
        import pandas
        print("[OK] Pandas")
    except ImportError:
        missing.append("pandas")
        print("[FAIL] Pandas")
    
    try:
        import torch
        print("[OK] PyTorch")
    except ImportError:
        missing.append("torch")
        print("[FAIL] PyTorch")
    
    try:
        import cv2
        print("[OK] OpenCV")
    except ImportError:
        missing.append("opencv-python")
        print("[FAIL] OpenCV")
    
    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n[OK] All dependencies installed!")
    return True

def create_folders():
    """Create necessary folders"""
    print("\nCreating folders...")
    
    folders = [
        'data/uploads',
        'data/processed',
        'data/sample_dicoms'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"[OK] {folder}")

def check_model():
    """Check if model file exists"""
    print("\nChecking model...")
    
    model_path = '../models/best_model.pth'
    if os.path.exists(model_path):
        print(f"[OK] Model found at {model_path}")
        return True
    else:
        print(f"[WARNING] Model not found at {model_path}")
        print("  The app will use a default pretrained model")
        return False

def start_app():
    """Start the Flask application"""
    print("\n" + "=" * 60)
    print("Starting Fetal Head Circumference Web App")
    print("=" * 60)
    
    try:
        from app import app
        print("\n[OK] App loaded successfully!")
        print("\n>> Starting server...")
        print("   Open your browser to: http://localhost:5000")
        print("   Press Ctrl+C to stop\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"\n[ERROR] Error starting app: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Fetal Head Circumference Web App - Startup")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create folders
    create_folders()
    
    # Check model
    check_model()
    
    # Start app
    start_app()
