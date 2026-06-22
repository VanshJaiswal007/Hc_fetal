"""
Model Inference Utilities
Handles loading trained model and running predictions
"""
import torch
import numpy as np
from PIL import Image
import cv2
import sys
import os

# Add parent directory to path to import model
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

try:
    from src.models.segmentation_models import SegmentationModelFactory, create_fetal_head_model
    USE_ADVANCED_MODEL = True
except ImportError as e:
    print(f"Warning: Advanced models not available, using simple U-Net ({e})")
    USE_ADVANCED_MODEL = False

from src.utils.ellipse_fitting import EllipseFitter
from .simple_model import SimpleUNet

class ModelInference:
    def __init__(self, model_path, device=None):
        """Initialize model for inference"""
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.input_size = (512, 512)
        self.ellipse_fitter = EllipseFitter()
    
    def _load_model(self, model_path):
        """Load trained model"""
        try:
            model = None
            
            # Try to load from checkpoint first
            if model_path and os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    
                    if USE_ADVANCED_MODEL:
                        # Check if model architecture info is in checkpoint
                        if 'model_name' in checkpoint:
                            model_name = checkpoint['model_name']
                        else:
                            model_name = 'unet_resnet34'  # Default
                        
                        # Create model
                        model = create_fetal_head_model(model_name, pretrained=False)
                    else:
                        # Use simple model
                        model = SimpleUNet(in_channels=3, out_channels=1)
                    
                    # Load weights
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    print(f"[OK] Model loaded from {model_path}")
                    
                except Exception as e:
                    print(f"Warning: Could not load checkpoint: {str(e)}")
                    model = None
            
            # If no model loaded yet, create default
            if model is None:
                print(f"[WARNING] Model file not found or could not be loaded. Using default model.")
                if USE_ADVANCED_MODEL:
                    try:
                        model = create_fetal_head_model('unet_resnet34', pretrained=True)
                        print("[OK] Using pretrained U-Net with ResNet34")
                    except Exception as e:
                        print(f"Warning: Could not create advanced model: {str(e)}")
                        model = SimpleUNet(in_channels=3, out_channels=1)
                        print("[OK] Using simple U-Net")
                else:
                    model = SimpleUNet(in_channels=3, out_channels=1)
                    print("[OK] Using simple U-Net")
            
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input - matches training preprocessing exactly"""
        try:
            # Load image with cv2 (same as training dataset)
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_size = (img.shape[1], img.shape[0])  # (width, height)
            
            # Resize using cv2 (same as albumentations A.Resize)
            img_resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
            
            # Normalize to [0, 1]
            img_array = img_resized.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization (same as training)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_array = (img_array - mean) / std
            
            # Transpose to (C, H, W) format
            img_array = img_array.transpose(2, 0, 1)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            return img_tensor, original_size
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def predict(self, image_path, pixel_spacing):
        """Run prediction on image"""
        try:
            # Preprocess
            img_tensor, original_size = self.preprocess_image(image_path)
            img_tensor = img_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(img_tensor)
                pred_mask_prob = torch.sigmoid(output)
            
            # Convert to numpy (keep as probability for better resizing)
            pred_mask_prob = pred_mask_prob.cpu().numpy()[0, 0]
            
            # Resize probability mask back to original size
            pred_mask_prob_resized = cv2.resize(
                pred_mask_prob,
                original_size,
                interpolation=cv2.INTER_LINEAR
            )
            
            # Apply threshold after resizing
            pred_mask_binary = (pred_mask_prob_resized > 0.5).astype(np.float32)
            
            # Convert to uint8 for ellipse fitting (0-255 range)
            pred_mask_uint8 = (pred_mask_binary * 255).astype(np.uint8)
            
            # Fit ellipse using robust fitter
            ellipse_result = self.ellipse_fitter.process_mask(
                pred_mask_uint8, 
                pixel_size=pixel_spacing,
                method='robust'
            )
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence(pred_mask_prob_resized, pred_mask_binary)
            
            # Extract results
            if ellipse_result['success']:
                ellipse_params = ellipse_result['ellipse_parameters']
                hc_mm = ellipse_result['hc_measurements']['head_circumference_mm']
                hc_measurements = ellipse_result['hc_measurements']
            else:
                ellipse_params = None
                hc_mm = None
                hc_measurements = None
                print(f"Warning: Ellipse fitting failed - {ellipse_result['error_message']}")
            
            result = {
                'mask': pred_mask_binary,
                'mask_probability': pred_mask_prob_resized,
                'ellipse_params': ellipse_params,
                'head_circumference_mm': hc_mm,
                'hc_measurements': hc_measurements,
                'pixel_spacing': pixel_spacing,
                'original_size': original_size,
                'ellipse_fitting_success': ellipse_result['success'],
                'confidence': confidence_metrics
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")
    
    def _calculate_confidence(self, mask_prob, mask_binary):
        """Calculate confidence metrics for the prediction"""
        try:
            # Mean probability of predicted positive pixels
            positive_pixels = mask_binary > 0.5
            if np.sum(positive_pixels) > 0:
                mean_confidence = np.mean(mask_prob[positive_pixels])
            else:
                mean_confidence = 0.0
            
            # Overall prediction confidence (mean of all probabilities)
            overall_confidence = np.mean(mask_prob)
            
            # Prediction certainty (how far from 0.5 threshold)
            certainty = np.mean(np.abs(mask_prob - 0.5)) * 2  # Scale to 0-1
            
            # Segmentation quality score (higher is better)
            # Based on how confident the model is about its predictions
            quality_score = mean_confidence
            
            return {
                'mean_confidence': float(mean_confidence),
                'overall_confidence': float(overall_confidence),
                'certainty': float(certainty),
                'quality_score': float(quality_score),
                'confidence_percentage': float(mean_confidence * 100)
            }
        except Exception as e:
            print(f"Warning: Could not calculate confidence metrics: {e}")
            return {
                'mean_confidence': 0.0,
                'overall_confidence': 0.0,
                'certainty': 0.0,
                'quality_score': 0.0,
                'confidence_percentage': 0.0
            }
    
    def save_visualization(self, image_path, prediction_result, output_path):
        """Save visualization of prediction"""
        try:
            # Load original image with cv2 (same as preprocessing)
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get mask and ellipse
            mask = prediction_result['mask']
            ellipse_params = prediction_result['ellipse_params']
            hc_mm = prediction_result['head_circumference_mm']
            
            # Verify mask and image sizes match
            if mask.shape != (img_array.shape[0], img_array.shape[1]):
                raise ValueError(f"Mask shape {mask.shape} doesn't match image shape {img_array.shape[:2]}")
            
            # Create overlay
            overlay = img_array.copy()
            
            # Draw mask in semi-transparent red
            # Ensure mask is binary (0 or 1)
            mask_binary = (mask > 0.5).astype(np.uint8)
            mask_colored = np.zeros_like(img_array)
            mask_colored[mask_binary > 0] = [255, 0, 0]
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            
            # Draw ellipse if available
            if ellipse_params is not None:
                center = (int(ellipse_params['center_x']), int(ellipse_params['center_y']))
                # Use semi-axes directly (already in correct format)
                axes = (int(ellipse_params['semi_major_axis']), int(ellipse_params['semi_minor_axis']))
                angle = ellipse_params['angle_degrees']
                
                cv2.ellipse(overlay, center, axes, angle, 0, 360, (0, 255, 0), 2)
                
                # Add text with HC measurement
                y_offset = 30
                if hc_mm is not None:
                    text = f"HC: {hc_mm:.2f} mm"
                    cv2.putText(overlay, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    y_offset += 30
                    
                    # Add additional measurements if available
                    hc_measurements = prediction_result.get('hc_measurements')
                    if hc_measurements:
                        text2 = f"OFD: {hc_measurements['occipitofrontal_diameter_mm']:.1f} mm"
                        text3 = f"BPD: {hc_measurements['biparietal_diameter_mm']:.1f} mm"
                        cv2.putText(overlay, text2, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
                        cv2.putText(overlay, text3, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
            
            # Add confidence score
            confidence = prediction_result.get('confidence', {})
            if confidence.get('confidence_percentage') is not None:
                conf_text = f"Confidence: {confidence['confidence_percentage']:.1f}%"
                # Position at bottom left
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_y = overlay.shape[0] - 20
                
                # Add background rectangle for better visibility
                cv2.rectangle(overlay, (5, text_y - text_size[1] - 5), 
                            (text_size[0] + 15, text_y + 5), (0, 0, 0), -1)
                
                # Choose color based on confidence level
                if confidence['confidence_percentage'] >= 80:
                    color = (0, 255, 0)  # Green - high confidence
                elif confidence['confidence_percentage'] >= 60:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 165, 255)  # Orange - low confidence
                
                cv2.putText(overlay, conf_text, (10, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Save using cv2 (convert RGB back to BGR for cv2)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, overlay_bgr)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error saving visualization: {str(e)}")
