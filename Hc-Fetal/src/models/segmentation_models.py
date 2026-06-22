"""
State-of-the-art Segmentation Models for Fetal Head Circumference

This module implements various modern segmentation architectures:
- U-Net with ResNet, EfficientNet, and other backbones
- DeepLabV3+ with various encoders
- FPN (Feature Pyramid Network)
- LinkNet
- PSPNet
- Custom architectures optimized for medical imaging

All models are designed for binary segmentation of fetal head circumference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import segmentation_models_pytorch as smp
from torchvision import models
import logging

logger = logging.getLogger(__name__)

class SegmentationModelFactory:
    """
    Factory class for creating various segmentation models
    """
    
    AVAILABLE_MODELS = {
        'unet_resnet34': 'U-Net with ResNet34 backbone',
        'unet_resnet50': 'U-Net with ResNet50 backbone',
        'unet_resnet101': 'U-Net with ResNet101 backbone',
        'unet_efficientnet_b0': 'U-Net with EfficientNet-B0 backbone',
        'unet_efficientnet_b4': 'U-Net with EfficientNet-B4 backbone',
        'unet_efficientnet_b7': 'U-Net with EfficientNet-B7 backbone',
        'deeplabv3plus_resnet50': 'DeepLabV3+ with ResNet50',
        'deeplabv3plus_resnet101': 'DeepLabV3+ with ResNet101',
        'fpn_resnet50': 'Feature Pyramid Network with ResNet50',
        'linknet_resnet34': 'LinkNet with ResNet34',
        'pspnet_resnet50': 'PSPNet with ResNet50',
        'unetplusplus_resnet34': 'U-Net++ with ResNet34',
        'manet_resnet34': 'MA-Net with ResNet34',
        'custom_medical_unet': 'Custom U-Net optimized for medical imaging'
    }
    
    @classmethod
    def create_model(cls, 
                    model_name: str,
                    in_channels: int = 3,
                    classes: int = 1,
                    activation: Optional[str] = None,
                    encoder_weights: str = 'imagenet',
                    **kwargs) -> nn.Module:
        """
        Create a segmentation model
        
        Args:
            model_name: Name of the model to create
            in_channels: Number of input channels
            classes: Number of output classes
            activation: Activation function for output
            encoder_weights: Pretrained weights for encoder
            **kwargs: Additional model-specific parameters
            
        Returns:
            PyTorch model
        """
        if model_name not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.AVAILABLE_MODELS.keys())}")
        
        logger.info(f"Creating model: {model_name}")
        
        # U-Net variants
        if model_name.startswith('unet_'):
            encoder_name = model_name.replace('unet_', '')
            return smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
                **kwargs
            )
        
        # DeepLabV3+ variants
        elif model_name.startswith('deeplabv3plus_'):
            encoder_name = model_name.replace('deeplabv3plus_', '')
            return smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
                **kwargs
            )
        
        # FPN variants
        elif model_name.startswith('fpn_'):
            encoder_name = model_name.replace('fpn_', '')
            return smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
                **kwargs
            )
        
        # LinkNet variants
        elif model_name.startswith('linknet_'):
            encoder_name = model_name.replace('linknet_', '')
            return smp.Linknet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
                **kwargs
            )
        
        # PSPNet variants
        elif model_name.startswith('pspnet_'):
            encoder_name = model_name.replace('pspnet_', '')
            return smp.PSPNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
                **kwargs
            )
        
        # U-Net++ variants
        elif model_name.startswith('unetplusplus_'):
            encoder_name = model_name.replace('unetplusplus_', '')
            return smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
                **kwargs
            )
        
        # MA-Net variants
        elif model_name.startswith('manet_'):
            encoder_name = model_name.replace('manet_', '')
            return smp.MAnet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
                **kwargs
            )
        
        # Custom medical U-Net
        elif model_name == 'custom_medical_unet':
            return CustomMedicalUNet(
                in_channels=in_channels,
                classes=classes,
                **kwargs
            )
        
        else:
            raise NotImplementedError(f"Model {model_name} not implemented yet")
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """List all available models with descriptions"""
        return cls.AVAILABLE_MODELS.copy()


class CustomMedicalUNet(nn.Module):
    """
    Custom U-Net architecture optimized for medical imaging
    
    Features:
    - Residual connections in encoder blocks
    - Attention gates in decoder
    - Deep supervision
    - Dropout for regularization
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 classes: int = 1,
                 base_channels: int = 64,
                 depth: int = 4,
                 dropout_rate: float = 0.1,
                 use_attention: bool = True,
                 use_deep_supervision: bool = False):
        super().__init__()
        
        self.depth = depth
        self.use_attention = use_attention
        self.use_deep_supervision = use_deep_supervision
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        
        channels = [in_channels] + [base_channels * (2**i) for i in range(depth + 1)]
        
        for i in range(depth + 1):
            self.encoder_blocks.append(
                ResidualBlock(channels[i], channels[i + 1], dropout_rate)
            )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upconv_blocks = nn.ModuleList()
        
        if use_attention:
            self.attention_blocks = nn.ModuleList()
        
        for i in range(depth):
            # Upconvolution
            self.upconv_blocks.append(
                nn.ConvTranspose2d(channels[depth - i + 1], channels[depth - i], 2, 2)
            )
            
            # Attention gate
            if use_attention:
                self.attention_blocks.append(
                    AttentionGate(channels[depth - i], channels[depth - i + 1], channels[depth - i] // 2)
                )
            
            # Decoder block
            self.decoder_blocks.append(
                ResidualBlock(channels[depth - i] * 2, channels[depth - i], dropout_rate)
            )
        
        # Final convolution
        self.final_conv = nn.Conv2d(base_channels, classes, 1)
        
        # Deep supervision outputs
        if use_deep_supervision:
            self.deep_supervision_outputs = nn.ModuleList([
                nn.Conv2d(channels[depth - i], classes, 1) for i in range(depth)
            ])
    
    def forward(self, x):
        # Encoder
        encoder_features = []
        
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i < self.depth:
                encoder_features.append(x)
                x = self.pool(x)
        
        # Decoder
        deep_supervision_outputs = []
        
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_blocks, self.decoder_blocks)):
            x = upconv(x)
            
            # Get skip connection
            skip = encoder_features[self.depth - 1 - i]
            
            # Apply attention gate if enabled
            if self.use_attention:
                skip = self.attention_blocks[i](skip, x)
            
            # Concatenate and decode
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)
            
            # Deep supervision
            if self.use_deep_supervision and i < len(self.deep_supervision_outputs):
                ds_output = self.deep_supervision_outputs[i](x)
                deep_supervision_outputs.append(ds_output)
        
        # Final output
        output = self.final_conv(x)
        
        if self.use_deep_supervision and self.training:
            return output, deep_supervision_outputs
        else:
            return output


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant features"""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class FetalHeadSegmentationModel(nn.Module):
    """
    Wrapper class for fetal head segmentation with additional features
    """
    
    def __init__(self,
                 backbone_model: nn.Module,
                 use_tta: bool = False,
                 use_post_processing: bool = True):
        super().__init__()
        
        self.backbone = backbone_model
        self.use_tta = use_tta
        self.use_post_processing = use_post_processing
    
    def forward(self, x):
        if self.training or not self.use_tta:
            return self.backbone(x)
        else:
            return self._forward_with_tta(x)
    
    def _forward_with_tta(self, x):
        """Test Time Augmentation"""
        outputs = []
        
        # Original
        outputs.append(torch.sigmoid(self.backbone(x)))
        
        # Horizontal flip
        x_flip = torch.flip(x, dims=[3])
        out_flip = torch.sigmoid(self.backbone(x_flip))
        outputs.append(torch.flip(out_flip, dims=[3]))
        
        # Vertical flip
        x_vflip = torch.flip(x, dims=[2])
        out_vflip = torch.sigmoid(self.backbone(x_vflip))
        outputs.append(torch.flip(out_vflip, dims=[2]))
        
        # Average predictions
        return torch.mean(torch.stack(outputs), dim=0)


def create_fetal_head_model(model_name: str = 'unet_resnet34',
                           input_size: tuple = (512, 512),
                           pretrained: bool = True,
                           **kwargs) -> FetalHeadSegmentationModel:
    """
    Create a complete fetal head segmentation model
    
    Args:
        model_name: Name of the backbone model
        input_size: Input image size (H, W)
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model parameters
        
    Returns:
        Complete segmentation model
    """
    # Create backbone model
    backbone = SegmentationModelFactory.create_model(
        model_name=model_name,
        in_channels=3,
        classes=1,
        activation=None,  # We'll apply sigmoid in loss function
        encoder_weights='imagenet' if pretrained else None,
        **kwargs
    )
    
    # Wrap in fetal head model
    model = FetalHeadSegmentationModel(
        backbone_model=backbone,
        use_tta=kwargs.get('use_tta', False),
        use_post_processing=kwargs.get('use_post_processing', True)
    )
    
    logger.info(f"Created fetal head segmentation model: {model_name}")
    logger.info(f"Input size: {input_size}, Pretrained: {pretrained}")
    
    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': model.__class__.__name__
    }


def print_model_summary(model: nn.Module, input_size: tuple = (3, 512, 512)):
    """
    Print a summary of the model
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    try:
        from torchsummary import summary
        summary(model, input_size)
    except ImportError:
        info = get_model_info(model)
        print(f"Model: {info['architecture']}")
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Trainable parameters: {info['trainable_parameters']:,}")
        print(f"Model size: {info['model_size_mb']:.2f} MB")


if __name__ == "__main__":
    # Example usage
    print("Available segmentation models:")
    for name, desc in SegmentationModelFactory.list_models().items():
        print(f"  {name}: {desc}")
    
    # Create a model
    model = create_fetal_head_model('unet_resnet34')
    print(f"\nCreated model: {model}")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    
    # Print model info
    info = get_model_info(model)
    print(f"\nModel info: {info}")