"""
Convert HC18 PNG images to DICOM format
Includes pixel spacing and proper ultrasound metadata
"""
import pydicom
from pydicom.dataset import Dataset, FileDataset
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import os

def png_to_dicom(png_path, pixel_size_mm, output_path, patient_id=None, hc_mm=None):
    """
    Convert PNG ultrasound image to DICOM with pixel spacing
    
    Args:
        png_path: Path to PNG file
        pixel_size_mm: Pixel spacing in mm/pixel
        output_path: Path to save DICOM file
        patient_id: Optional patient ID
        hc_mm: Optional head circumference measurement
    """
    print(f"\nConverting: {png_path}")
    
    # Read PNG
    img = Image.open(png_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    
    print(f"  Image size: {img_array.shape}")
    print(f"  Pixel spacing: {pixel_size_mm} mm/pixel")
    
    # Create file meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.6.1'  # Ultrasound Image Storage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    
    # Create the FileDataset instance
    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Add required DICOM tags
    ds.PatientName = patient_id if patient_id else "HC18_Patient"
    ds.PatientID = patient_id if patient_id else os.path.basename(png_path).split('.')[0]
    
    # Study information
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.now().strftime('%H%M%S')
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    
    # Modality
    ds.Modality = 'US'  # Ultrasound
    
    # Manufacturer information
    ds.Manufacturer = 'HC18 Dataset'
    ds.ManufacturerModelName = 'Fetal Ultrasound'
    ds.InstitutionName = 'Research Dataset'
    
    # Image information
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = img_array.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    
    # THE IMPORTANT PART: Pixel Spacing
    # [row spacing, column spacing] in mm
    ds.PixelSpacing = [pixel_size_mm, pixel_size_mm]
    
    # Window/Level for display
    ds.WindowCenter = [127]
    ds.WindowWidth = [255]
    
    # Add head circumference as a comment if provided
    if hc_mm is not None:
        ds.ImageComments = f"Head Circumference: {hc_mm} mm"
    
    # Set pixel data
    ds.PixelData = img_array.tobytes()
    
    # Save DICOM file
    ds.save_as(output_path, write_like_original=False)
    print(f"  ✓ Saved: {output_path}")
    print(f"  DICOM Pixel Spacing: {ds.PixelSpacing} mm")
    
    return output_path

def convert_hc18_samples():
    """Convert sample images from HC18 dataset to DICOM"""
    
    # Load metadata CSV
    csv_path = 'hc18_dataset/training_set_pixel_size_and_HC.csv'
    df = pd.read_csv(csv_path)
    
    # Create output folder
    output_folder = 'webapp/data/sample_dicoms'
    os.makedirs(output_folder, exist_ok=True)
    
    # Select 3 sample images
    sample_files = ['1_HC.png', '2_HC.png', '3_HC.png']
    
    print("=" * 60)
    print("Converting HC18 PNG images to DICOM format")
    print("=" * 60)
    
    converted_files = []
    
    for filename in sample_files:
        # Get metadata from CSV
        row = df[df['filename'] == filename]
        
        if row.empty:
            print(f"Warning: {filename} not found in CSV")
            continue
        
        pixel_size = row['pixel size'].values[0]
        hc_mm = row['head circumference (mm)'].values[0]
        
        # Paths
        png_path = f'hc18_dataset/training_set/training_set/{filename}'
        dcm_filename = filename.replace('.png', '.dcm')
        dcm_path = os.path.join(output_folder, dcm_filename)
        
        if not os.path.exists(png_path):
            print(f"Warning: {png_path} not found")
            continue
        
        # Convert
        try:
            output = png_to_dicom(
                png_path=png_path,
                pixel_size_mm=pixel_size,
                output_path=dcm_path,
                patient_id=filename.split('.')[0],
                hc_mm=hc_mm
            )
            converted_files.append(output)
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"Conversion complete! {len(converted_files)} files created")
    print("=" * 60)
    
    # Verify the DICOM files
    print("\nVerifying DICOM files...")
    for dcm_path in converted_files:
        verify_dicom(dcm_path)
    
    return converted_files

def verify_dicom(dcm_path):
    """Verify DICOM file and display key information"""
    try:
        ds = pydicom.dcmread(dcm_path)
        
        print(f"\n📄 {os.path.basename(dcm_path)}")
        print(f"  Patient ID: {ds.PatientID}")
        print(f"  Modality: {ds.Modality}")
        print(f"  Image Size: {ds.Rows} x {ds.Columns}")
        print(f"  Pixel Spacing: {ds.PixelSpacing} mm")
        print(f"  Bits Allocated: {ds.BitsAllocated}")
        
        if hasattr(ds, 'ImageComments'):
            print(f"  Comments: {ds.ImageComments}")
        
        print(f"  ✓ Valid DICOM file")
        
    except Exception as e:
        print(f"  ✗ Error reading DICOM: {str(e)}")

def test_dicom_to_png_conversion(dcm_path):
    """Test converting DICOM back to PNG to verify data integrity"""
    print(f"\n🔄 Testing round-trip conversion for: {os.path.basename(dcm_path)}")
    
    try:
        # Read DICOM
        ds = pydicom.dcmread(dcm_path)
        pixel_array = ds.pixel_array
        
        # Save as PNG
        test_png_path = dcm_path.replace('.dcm', '_test.png')
        img = Image.fromarray(pixel_array)
        img.save(test_png_path)
        
        print(f"  ✓ Successfully converted back to PNG: {os.path.basename(test_png_path)}")
        print(f"  Pixel spacing preserved: {ds.PixelSpacing} mm")
        
        return test_png_path
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return None

if __name__ == '__main__':
    # Convert sample images
    converted_files = convert_hc18_samples()
    
    # Test round-trip conversion on first file
    if converted_files:
        print("\n" + "=" * 60)
        print("Testing round-trip conversion (DICOM → PNG)")
        print("=" * 60)
        test_dicom_to_png_conversion(converted_files[0])
    
    print("\n" + "=" * 60)
    print("✨ All done!")
    print("=" * 60)
    print(f"\nDICOM files saved in: webapp/data/sample_dicoms/")
    print("\nYou can now:")
    print("1. Test these DICOM files in the web app")
    print("2. Verify pixel spacing extraction")
    print("3. Test DICOM to PNG conversion")
