"""
Test script to verify webapp setup
"""
import os
import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import flask
        print("✓ Flask")
    except ImportError as e:
        print(f"✗ Flask: {e}")
    
    try:
        import pydicom
        print("✓ pydicom")
    except ImportError as e:
        print(f"✗ pydicom: {e}")
    
    try:
        from PIL import Image
        print("✓ Pillow")
    except ImportError as e:
        print(f"✗ Pillow: {e}")
    
    try:
        import numpy
        print("✓ NumPy")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
    
    try:
        import pandas
        print("✓ Pandas")
    except ImportError as e:
        print(f"✗ Pandas: {e}")
    
    try:
        import torch
        print("✓ PyTorch")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
    
    try:
        import cv2
        print("✓ OpenCV")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")

def test_folders():
    """Test if required folders exist"""
    print("\nTesting folders...")
    
    folders = [
        'data/uploads',
        'data/processed',
        'templates',
        'utils'
    ]
    
    for folder in folders:
        if os.path.exists(folder):
            print(f"✓ {folder}")
        else:
            print(f"✗ {folder} (will be created)")
            os.makedirs(folder, exist_ok=True)

def test_model():
    """Test if model file exists"""
    print("\nTesting model...")
    
    model_path = '../models/best_model.pth'
    if os.path.exists(model_path):
        print(f"✓ Model found at {model_path}")
    else:
        print(f"⚠ Model not found at {model_path}")
        print("  You'll need to train a model or update the path in app.py")

def test_utils():
    """Test if utility modules can be imported"""
    print("\nTesting utility modules...")
    
    try:
        from utils.dicom_processor import DICOMProcessor
        print("✓ DICOMProcessor")
    except ImportError as e:
        print(f"✗ DICOMProcessor: {e}")
    
    try:
        from utils.image_processor import ImageProcessor
        print("✓ ImageProcessor")
    except ImportError as e:
        print(f"✗ ImageProcessor: {e}")
    
    try:
        from utils.model_inference import ModelInference
        print("✓ ModelInference")
    except ImportError as e:
        print(f"✗ ModelInference: {e}")

if __name__ == '__main__':
    print("=" * 50)
    print("Fetal Head Circumference Web App - Setup Test")
    print("=" * 50)
    
    test_imports()
    test_folders()
    test_model()
    test_utils()
    
    print("\n" + "=" * 50)
    print("Setup test complete!")
    print("=" * 50)
    print("\nTo start the web app, run:")
    print("  python app.py")
