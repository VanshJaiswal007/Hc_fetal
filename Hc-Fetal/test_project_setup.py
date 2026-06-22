"""
Comprehensive test script for HC-Fetal project
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_core_imports():
    """Test core Python imports"""
    print("=" * 60)
    print("TESTING CORE PYTHON IMPORTS")
    print("=" * 60)
    
    imports_to_test = [
        ('os', 'os'),
        ('sys', 'sys'),
        ('json', 'json'),
        ('datetime', 'datetime'),
        ('uuid', 'uuid'),
    ]
    
    for module_name, _ in imports_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
    
    return True

def test_optional_imports():
    """Test optional dependencies"""
    print("\n" + "=" * 60)
    print("TESTING OPTIONAL DEPENDENCIES")
    print("=" * 60)
    
    optional_deps = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('pandas', 'Pandas'),
        ('torch', 'PyTorch'),
        ('flask', 'Flask'),
        ('pydicom', 'pyDICOM'),
        ('reportlab', 'ReportLab'),
    ]
    
    results = {}
    for module_name, friendly_name in optional_deps:
        try:
            __import__(module_name)
            print(f"✓ {friendly_name} ({module_name})")
            results[module_name] = True
        except ImportError as e:
            print(f"✗ {friendly_name} ({module_name}): {e}")
            results[module_name] = False
    
    return results

def test_src_modules():
    """Test src package modules"""
    print("\n" + "=" * 60)
    print("TESTING SRC PACKAGE MODULES")
    print("=" * 60)
    
    try:
        from src.utils.ellipse_fitting import EllipseFitter, fit_ellipse_to_mask
        print("✓ src.utils.ellipse_fitting (EllipseFitter, fit_ellipse_to_mask)")
    except ImportError as e:
        print(f"✗ src.utils.ellipse_fitting: {e}")
        return False
    
    try:
        from src.models.segmentation_models import create_fetal_head_model
        print("✓ src.models.segmentation_models (create_fetal_head_model)")
    except ImportError as e:
        print(f"✗ src.models.segmentation_models: {e}")
        return False
    
    return True

def test_webapp_utils():
    """Test webapp utility modules"""
    print("\n" + "=" * 60)
    print("TESTING WEBAPP UTILITY MODULES")
    print("=" * 60)
    
    os.chdir('webapp')
    
    try:
        from utils.dicom_processor import DICOMProcessor
        print("✓ utils.dicom_processor (DICOMProcessor)")
    except ImportError as e:
        print(f"✗ utils.dicom_processor: {e}")
    
    try:
        from utils.image_processor import ImageProcessor
        print("✓ utils.image_processor (ImageProcessor)")
    except ImportError as e:
        print(f"✗ utils.image_processor: {e}")
    
    try:
        from utils.csv_exporter import ResultsCSVExporter
        print("✓ utils.csv_exporter (ResultsCSVExporter)")
    except ImportError as e:
        print(f"✗ utils.csv_exporter: {e}")
    
    try:
        from utils.pdf_generator import FetalHCReportGenerator
        print("✓ utils.pdf_generator (FetalHCReportGenerator)")
    except ImportError as e:
        print(f"✗ utils.pdf_generator: {e}")
    
    try:
        from utils.simple_model import SimpleUNet
        print("✓ utils.simple_model (SimpleUNet)")
    except ImportError as e:
        print(f"✗ utils.simple_model: {e}")
    
    try:
        from utils.model_inference import ModelInference
        print("✓ utils.model_inference (ModelInference)")
    except ImportError as e:
        print(f"✗ utils.model_inference: {e}")
    
    os.chdir('..')
    return True

def test_directories():
    """Test if required directories exist"""
    print("\n" + "=" * 60)
    print("TESTING DIRECTORY STRUCTURE")
    print("=" * 60)
    
    dirs = [
        'src',
        'src/models',
        'src/utils',
        'src/training',
        'webapp',
        'webapp/utils',
        'webapp/templates',
        'webapp/data',
    ]
    
    for dir_path in dirs:
        if os.path.isdir(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path}")
    
    return True

def test_init_files():
    """Test if __init__.py files exist"""
    print("\n" + "=" * 60)
    print("TESTING __init__.py FILES")
    print("=" * 60)
    
    init_files = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/utils/__init__.py',
        'src/training/__init__.py',
        'webapp/utils/__init__.py',
    ]
    
    for init_file in init_files:
        if os.path.exists(init_file):
            print(f"✓ {init_file}")
        else:
            print(f"✗ {init_file}")
    
    return True

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "HC-FETAL PROJECT VERIFICATION" + " " * 13 + "║")
    print("╚" + "=" * 58 + "╝")
    
    test_core_imports()
    optional = test_optional_imports()
    test_directories()
    test_init_files()
    
    # Only test src modules if dependencies are available
    if optional.get('torch') and optional.get('cv2'):
        test_src_modules()
        if optional.get('flask'):
            test_webapp_utils()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    dependencies_status = {
        'torch': optional.get('torch', False),
        'cv2': optional.get('cv2', False),
        'flask': optional.get('flask', False),
        'pydicom': optional.get('pydicom', False),
        'reportlab': optional.get('reportlab', False),
    }
    
    if all(dependencies_status.values()):
        print("✓ All dependencies installed!")
        print("✓ Project is ready to run")
        print("\nTo start the web app:")
        print("  cd webapp")
        print("  python app.py")
    else:
        print("⚠ Some dependencies are missing")
        missing = [k for k, v in dependencies_status.items() if not v]
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nTo install required dependencies:")
        print("  pip install -r requirements.txt")
    
    print("\n")

if __name__ == '__main__':
    main()
