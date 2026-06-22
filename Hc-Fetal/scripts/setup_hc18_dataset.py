#!/usr/bin/env python3
"""
HC18 Dataset Setup Script

This script demonstrates the complete workflow for setting up the HC18 dataset:
1. Dataset analysis and validation
2. Data organization and preprocessing
3. Train/validation split creation
4. Data loader setup
5. Visualization of dataset statistics

Usage:
    python scripts/setup_hc18_dataset.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.hc18_dataset_handler import HC18DatasetHandler, create_data_loaders
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Setup HC18 dataset for training")
    parser.add_argument("--dataset_root", type=str, default="hc18_dataset",
                       help="Path to HC18 dataset root directory")
    parser.add_argument("--output_dir", type=str, default="data/processed/hc18",
                       help="Output directory for organized dataset")
    parser.add_argument("--val_size", type=float, default=0.2,
                       help="Validation set size (fraction)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for data loaders")
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512],
                       help="Target image size (height width)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--skip_organization", action="store_true",
                       help="Skip dataset organization (use existing)")
    parser.add_argument("--create_visualization", action="store_true",
                       help="Create dataset analysis visualization")
    
    args = parser.parse_args()
    
    logger.info("Starting HC18 dataset setup...")
    
    # Initialize dataset handler
    try:
        handler = HC18DatasetHandler(args.dataset_root)
        logger.info("✓ Dataset handler initialized successfully")
    except FileNotFoundError as e:
        logger.error(f"✗ Dataset not found: {e}")
        logger.error("Please ensure the HC18 dataset is downloaded and extracted to the correct location")
        return 1
    
    # Analyze dataset
    logger.info("Analyzing dataset...")
    analysis = handler.analyze_dataset()
    handler.print_dataset_summary()
    
    # Create visualization if requested
    if args.create_visualization:
        logger.info("Creating dataset analysis visualization...")
        viz_path = Path(args.output_dir) / "hc18_analysis.png"
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        handler.visualize_dataset_analysis(str(viz_path))
        logger.info(f"✓ Visualization saved to {viz_path}")
    
    # Organize dataset
    if not args.skip_organization:
        logger.info("Organizing dataset...")
        organized_path = handler.organize_dataset(args.output_dir)
        logger.info(f"✓ Dataset organized in {organized_path}")
    else:
        organized_path = Path(args.output_dir)
        if not organized_path.exists():
            logger.error(f"✗ Organized dataset not found at {organized_path}")
            return 1
        logger.info(f"Using existing organized dataset at {organized_path}")
    
    # Create train/validation split
    logger.info("Creating train/validation split...")
    train_df, val_df = handler.create_train_val_split(
        str(organized_path), 
        val_size=args.val_size,
        stratify_by_mask=True
    )
    logger.info("✓ Train/validation split created")
    
    # Calculate image statistics
    logger.info("Calculating image statistics...")
    stats = handler.get_image_statistics(str(organized_path))
    logger.info(f"✓ Image statistics calculated from {stats['sample_size']} samples")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    try:
        train_loader, val_loader = create_data_loaders(
            str(organized_path),
            batch_size=args.batch_size,
            image_size=tuple(args.image_size),
            num_workers=args.num_workers
        )
        logger.info(f"✓ Data loaders created:")
        logger.info(f"  Training batches: {len(train_loader)}")
        logger.info(f"  Validation batches: {len(val_loader)}")
        
        # Test data loading
        logger.info("Testing data loading...")
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        logger.info(f"✓ Data loading test successful:")
        logger.info(f"  Train batch - Images: {train_batch['image'].shape}, Masks: {train_batch['mask'].shape}")
        logger.info(f"  Val batch - Images: {val_batch['image'].shape}, Masks: {val_batch['mask'].shape}")
        
    except Exception as e:
        logger.error(f"✗ Error creating data loaders: {e}")
        return 1
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("HC18 DATASET SETUP COMPLETE")
    logger.info("="*60)
    logger.info(f"Dataset location: {organized_path}")
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Samples with masks: {train_df['has_mask'].sum() + val_df['has_mask'].sum()}")
    logger.info(f"Samples without masks: {(~train_df['has_mask']).sum() + (~val_df['has_mask']).sum()}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("="*60)
    
    # Save setup configuration
    config = {
        'dataset_root': args.dataset_root,
        'organized_path': str(organized_path),
        'val_size': args.val_size,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'num_workers': args.num_workers,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'samples_with_masks': int(train_df['has_mask'].sum() + val_df['has_mask'].sum()),
        'samples_without_masks': int((~train_df['has_mask']).sum() + (~val_df['has_mask']).sum()),
        'image_statistics': stats,
        'dataset_analysis': analysis
    }
    
    import json
    config_path = organized_path / "setup_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Setup configuration saved to {config_path}")
    logger.info("Dataset is ready for training!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)