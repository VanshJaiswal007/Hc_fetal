"""
Comprehensive Training Pipeline for Fetal Head Circumference Segmentation

This module provides:
- Advanced loss functions for medical segmentation
- Training and validation loops
- Metrics calculation and logging
- Model checkpointing and early stopping
- Learning rate scheduling
- Mixed precision training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
from tqdm import tqdm
import wandb
from collections import defaultdict
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class SegmentationLoss(nn.Module):
    """
    Combined loss function for boundary/contour segmentation
    
    For boundary segmentation with extreme class imbalance:
    - Dice loss helps learn the shape/overlap
    - BCE with pos_weight handles class imbalance
    - Focal loss down-weights easy examples
    """
    
    def __init__(self, 
                 loss_weights: Dict[str, float] = None,
                 smooth: float = 1e-6,
                 device: str = 'cuda'):
        super().__init__()
        
        self.smooth = smooth
        self.device = device
        
        # For boundary segmentation: Combine BCE + Dice + Focal
        # Dice helps learn the shape, BCE+Focal handle class imbalance
        self.loss_weights = loss_weights or {
            'bce': 0.3,
            'dice': 0.4,
            'focal': 0.3,
            'tversky': 0.0
        }
        
        # Use weighted BCE for class imbalance (boundary pixels are ~0.6% of total)
        # Weight positive class much higher - move to device
        pos_weight = torch.tensor([150.0]).to(device)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice loss for segmentation"""
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice_score
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
        """
        Focal loss for handling extreme class imbalance in boundary segmentation
        
        For boundary detection with ~0.6% positive pixels:
        - alpha=0.75: Focus more on the rare positive (boundary) class
        - gamma=2.0: Down-weight easy examples
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
    
    def tversky_loss(self, pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7) -> torch.Tensor:
        """Tversky loss - generalization of Dice loss"""
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        true_pos = (pred_flat * target_flat).sum()
        false_neg = (target_flat * (1 - pred_flat)).sum()
        false_pos = ((1 - target_flat) * pred_flat).sum()
        
        tversky_index = (true_pos + self.smooth) / (true_pos + alpha * false_neg + beta * false_pos + self.smooth)
        
        return 1 - tversky_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            pred: Predicted logits [B, 1, H, W]
            target: Ground truth masks [B, H, W] or [B, 1, H, W]
            
        Returns:
            Dictionary with individual and total losses
        """
        # Ensure target has correct shape
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        losses = {}
        total_loss = 0
        
        # BCE Loss
        if self.loss_weights['bce'] > 0:
            losses['bce'] = self.bce_loss(pred, target.float())
            total_loss += self.loss_weights['bce'] * losses['bce']
        
        # Dice Loss
        if self.loss_weights['dice'] > 0:
            losses['dice'] = self.dice_loss(pred, target.float())
            total_loss += self.loss_weights['dice'] * losses['dice']
        
        # Focal Loss
        if self.loss_weights['focal'] > 0:
            losses['focal'] = self.focal_loss(pred, target.float())
            total_loss += self.loss_weights['focal'] * losses['focal']
        
        # Tversky Loss
        if self.loss_weights['tversky'] > 0:
            losses['tversky'] = self.tversky_loss(pred, target.float())
            total_loss += self.loss_weights['tversky'] * losses['tversky']
        
        losses['total'] = total_loss
        return losses


class SegmentationMetrics:
    """
    Comprehensive metrics for segmentation evaluation
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth
    
    def dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Dice coefficient (returns 0-100 percentage)"""
        # Ensure pred is sigmoid activated
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        pred_binary = (pred > self.threshold).float()
        target_binary = (target > self.threshold).float()
        
        intersection = (pred_binary * target_binary).sum()
        dice = (2. * intersection + self.smooth) / (pred_binary.sum() + target_binary.sum() + self.smooth)
        
        return dice.item() * 100.0  # Return as percentage
    
    def iou_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Intersection over Union (returns 0-100 percentage)"""
        # Ensure pred is sigmoid activated
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        pred_binary = (pred > self.threshold).float()
        target_binary = (target > self.threshold).float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou.item() * 100.0  # Return as percentage
    
    def pixel_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate pixel-wise accuracy (returns 0-100 percentage)"""
        # Ensure pred is sigmoid activated
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        pred_binary = (pred > self.threshold).float()
        target_binary = (target > self.threshold).float()
        
        correct = (pred_binary == target_binary).float().sum()
        total = target_binary.numel()
        
        accuracy = (correct / total).item()
        return accuracy * 100.0  # Return as percentage (0-100)
    
    def sensitivity_specificity(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        """Calculate sensitivity and specificity (returns 0-100 percentages)"""
        # Ensure pred is sigmoid activated
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        pred_binary = (pred > self.threshold).float()
        target_binary = (target > self.threshold).float()
        
        tp = (pred_binary * target_binary).sum()
        tn = ((1 - pred_binary) * (1 - target_binary)).sum()
        fp = (pred_binary * (1 - target_binary)).sum()
        fn = ((1 - pred_binary) * target_binary).sum()
        
        sensitivity = (tp + self.smooth) / (tp + fn + self.smooth)
        specificity = (tn + self.smooth) / (tn + fp + self.smooth)
        
        return sensitivity.item() * 100.0, specificity.item() * 100.0  # Return as percentages
    
    def calculate_all_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Calculate all metrics (handles both single samples and batches)
        
        Args:
            pred: Model predictions [B, 1, H, W] or [1, H, W] (logits or probabilities)
            target: Ground truth masks [B, H, W] or [H, W] (0-1 range)
            
        Returns:
            Dictionary of metrics (all in 0-100 percentage range)
        """
        # If batch, calculate per-sample and average
        if pred.dim() == 4:  # Batch: [B, 1, H, W]
            batch_size = pred.shape[0]
            metrics_list = []
            
            for i in range(batch_size):
                sample_pred = pred[i]  # [1, H, W]
                sample_target = target[i]  # [H, W]
                
                sample_metrics = {
                    'dice': self.dice_coefficient(sample_pred, sample_target),
                    'iou': self.iou_score(sample_pred, sample_target),
                    'pixel_accuracy': self.pixel_accuracy(sample_pred, sample_target)
                }
                
                sensitivity, specificity = self.sensitivity_specificity(sample_pred, sample_target)
                sample_metrics['sensitivity'] = sensitivity
                sample_metrics['specificity'] = specificity
                
                metrics_list.append(sample_metrics)
            
            # Average across batch
            metrics = {}
            for key in metrics_list[0].keys():
                metrics[key] = sum(m[key] for m in metrics_list) / batch_size
            
            return metrics
        
        else:  # Single sample: [1, H, W]
            metrics = {
                'dice': self.dice_coefficient(pred, target),
                'iou': self.iou_score(pred, target),
                'pixel_accuracy': self.pixel_accuracy(pred, target)
            }
            
            sensitivity, specificity = self.sensitivity_specificity(pred, target)
            metrics['sensitivity'] = sensitivity
            metrics['specificity'] = specificity
            
            return metrics


class FetalHeadTrainer:
    """
    Comprehensive trainer for fetal head segmentation
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 save_dir: str = 'checkpoints',
                 use_amp: bool = True,
                 log_wandb: bool = False):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp
        self.log_wandb = log_wandb
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Initialize metrics calculator
        self.metrics_calculator = SegmentationMetrics()
        
        # Training history
        self.history = defaultdict(list)
        
        # Best model tracking (using IoU for boundary segmentation)
        self.best_val_iou = 0.0
        self.best_model_path = None
        
        logger.info(f"Trainer initialized with device: {device}")
        logger.info(f"Mixed precision training: {use_amp}")
        logger.info(f"Wandb logging: {log_wandb}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    losses = self.loss_fn(outputs, masks)
                
                # Backward pass
                self.scaler.scale(losses['total']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                losses = self.loss_fn(outputs, masks)
                losses['total'].backward()
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                batch_metrics = self.metrics_calculator.calculate_all_metrics(outputs, masks)
            
            # Store losses and metrics
            for key, value in losses.items():
                epoch_losses[key].append(value.item())
            
            for key, value in batch_metrics.items():
                epoch_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'acc': f"{batch_metrics['pixel_accuracy']:.1f}%",
                'iou': f"{batch_metrics['iou']:.1f}%"
            })
        
        # Calculate epoch averages
        epoch_results = {}
        for key, values in epoch_losses.items():
            epoch_results[f'train_{key}_loss'] = np.mean(values)
        
        for key, values in epoch_metrics.items():
            epoch_results[f'train_{key}'] = np.mean(values)
        
        return epoch_results
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {epoch}')
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        losses = self.loss_fn(outputs, masks)
                else:
                    outputs = self.model(images)
                    losses = self.loss_fn(outputs, masks)
                
                # Calculate metrics
                batch_metrics = self.metrics_calculator.calculate_all_metrics(outputs, masks)
                
                # Store losses and metrics
                for key, value in losses.items():
                    epoch_losses[key].append(value.item())
                
                for key, value in batch_metrics.items():
                    epoch_metrics[key].append(value)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'acc': f"{batch_metrics['pixel_accuracy']:.1f}%",
                    'iou': f"{batch_metrics['iou']:.1f}%"
                })
        
        # Calculate epoch averages
        epoch_results = {}
        for key, values in epoch_losses.items():
            epoch_results[f'val_{key}_loss'] = np.mean(values)
        
        for key, values in epoch_metrics.items():
            epoch_results[f'val_{key}'] = np.mean(values)
        
        return epoch_results
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': dict(self.history)
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            logger.info(f"New best model saved with val_iou: {metrics.get('val_iou', 0):.2f}%")
    
    def train(self, 
              num_epochs: int,
              early_stopping_patience: int = 10,
              save_every: int = 5) -> Dict[str, List[float]]:
        """
        Complete training loop
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Epochs to wait before early stopping
            save_every: Save checkpoint every N epochs
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_iou = 0.0
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train epoch
            train_results = self.train_epoch(epoch)
            
            # Validate epoch
            val_results = self.validate_epoch(epoch)
            
            # Combine results
            epoch_results = {**train_results, **val_results}
            
            # Update history
            for key, value in epoch_results.items():
                self.history[key].append(value)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['val_total_loss'])
                else:
                    self.scheduler.step()
            
            # Check for best model (using IoU for boundary segmentation)
            current_val_iou = val_results['val_iou']
            is_best = current_val_iou > best_val_iou
            
            if is_best:
                best_val_iou = current_val_iou
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, epoch_results, is_best)
            
            # Log results
            epoch_time = time.time() - start_time
            
            logger.info(f"Epoch {epoch}/{num_epochs} - {epoch_time:.2f}s")
            logger.info(f"Train Loss: {train_results['train_total_loss']:.4f}, Train Acc: {train_results['train_pixel_accuracy']:.2f}%, Train IoU: {train_results['train_iou']:.2f}%")
            logger.info(f"Val Loss: {val_results['val_total_loss']:.4f}, Val Acc: {val_results['val_pixel_accuracy']:.2f}%, Val IoU: {val_results['val_iou']:.2f}%")
            
            # Wandb logging
            if self.log_wandb:
                wandb.log({
                    'epoch': epoch,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time,
                    **epoch_results
                })
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        logger.info(f"Training completed. Best val_iou: {best_val_iou:.2f}%")
        return dict(self.history)
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_total_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_total_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_pixel_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(self.history['val_pixel_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Pixel Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Dice score plot
        axes[1, 0].plot(self.history['train_dice'], label='Train Dice')
        axes[1, 0].plot(self.history['val_dice'], label='Val Dice')
        axes[1, 0].set_title('Dice Coefficient')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # IoU plot
        axes[1, 1].plot(self.history['train_iou'], label='Train IoU')
        axes[1, 1].plot(self.history['val_iou'], label='Val IoU')
        axes[1, 1].set_title('Intersection over Union')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('IoU Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()


def create_optimizer(model: nn.Module, 
                    optimizer_name: str = 'adamw',
                    learning_rate: float = 1e-4,
                    weight_decay: float = 1e-4,
                    **kwargs) -> torch.optim.Optimizer:
    """Create optimizer"""
    
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_name: str = 'cosine',
                    num_epochs: int = 100,
                    **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler"""
    
    if scheduler_name.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, **kwargs)
    elif scheduler_name.lower() == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, **kwargs)
    elif scheduler_name.lower() == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, **kwargs)
    elif scheduler_name.lower() == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


if __name__ == "__main__":
    # Example usage
    print("Training utilities for fetal head segmentation ready!")
    
    # Test loss function
    loss_fn = SegmentationLoss()
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 256, 256)).float()
    
    losses = loss_fn(pred, target)
    print(f"Loss test: {losses}")
    
    # Test metrics
    metrics_calc = SegmentationMetrics()
    metrics = metrics_calc.calculate_all_metrics(torch.sigmoid(pred), target)
    print(f"Metrics test: {metrics}")