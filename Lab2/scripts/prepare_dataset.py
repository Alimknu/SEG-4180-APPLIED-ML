"""Download and prepare satellite building segmentation dataset"""

import os, json, numpy as np
from PIL import Image
from pathlib import Path
import logging
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    def __init__(self, output_dir='data/satellite_dataset', force_download=False):
        self.output_dir = Path(output_dir)
        self.force_download = force_download
        self.dataset = None
        # Create output directories
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'masks']:
                (self.output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    def download_satellite_dataset(self):
        """Download satellite building segmentation dataset"""
        logger.info("Downloading satellite building segmentation dataset...")
        try:
            # Load from Hugging Face Hub
            self.dataset = load_dataset(
                "keremberke/satellite-building-segmentation",
                name="full"
            )
            logger.info(f"Dataset loaded with splits: {self.dataset.keys()}")
            logger.info(f"Training samples: {len(self.dataset['train'])}")
            return True
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return False
    
    def make_mask(self, bbox, image_width, image_height):
        """Create binary mask from bounding box (Week 7 pixel mask generation)"""
        x_min, y_min, width, height = [int(v) for v in bbox]
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        x_max = min(x_min + width, image_width)
        y_max = min(y_min + height, image_height)
        mask[y_min:y_max, x_min:x_max] = 1
        return mask
    
    def create_composite_mask(self, example):
        """
        Create composite mask from all building bboxes
        
        Args:
            example: Dataset example with image and bboxes
            
        Returns:
            Composite mask
        """
        image = example['image']
        width, height = image.size
        
        # Create empty mask
        composite_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Add all building masks
        for bbox in example['objects']['bbox']:
            mask = self.make_mask(bbox, width, height)
            composite_mask = np.maximum(composite_mask, mask)
        
        return composite_mask
    
    def prepare_split(self, split_name='train', test_size=0.15, seed=42):
        """Split data into train/val/test and save to disk"""
        if self.dataset is None:
            logger.error("Dataset not loaded. Call download_satellite_dataset first.")
            return
        
        logger.info(f"Processing {split_name} split...")
        
        split_data = self.dataset[split_name]
        indices = list(range(len(split_data)))
        
        # Split into train/val/test
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed
        )
        
        val_size = int(len(train_idx) * 0.15)
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_size,
            random_state=seed
        )
        
        splits = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
        
        # Process each split
        for split_type, idx_list in splits.items():
            logger.info(f"Processing {split_type} split ({len(idx_list)} samples)...")
            
            for idx, sample_idx in enumerate(tqdm(idx_list, desc=f"{split_type} split")):
                example = split_data[sample_idx]
                
                # Save image
                image = example['image']
                image_name = f"{split_type}_{idx:04d}.jpg"
                image_path = self.output_dir / split_type / 'images' / image_name
                image.save(image_path, quality=95)
                
                # Create and save mask
                mask = self.create_composite_mask(example)
                mask_name = f"{split_type}_{idx:04d}.png"
                mask_path = self.output_dir / split_type / 'masks' / mask_name
                Image.fromarray(mask * 255, mode='L').save(mask_path)
                
                # Log sample info
                if idx < 3:
                    logger.info(f"  Sample {idx}: Image {image.size}, "
                               f"{len(example['objects']['bbox'])} buildings")
    
    def prepare_dataset(self):
        """Download satellite dataset and prepare train/val/test splits"""
        
        # Download dataset
        if not self.download_satellite_dataset():
            logger.error("Failed to download dataset")
            return False
        
        # Prepare splits
        self.prepare_split('train')
        if 'test' in self.dataset:
            self.prepare_split('test')
        
        # Save dataset metadata
        metadata = {
            'dataset_name': 'Satellite Building Segmentation',
            'source': 'https://huggingface.co/datasets/keremberke/satellite-building-segmentation',
            'task': 'semantic_segmentation',
            'num_classes': 2,
            'class_names': ['background', 'building'],
            'image_format': 'jpg',
            'mask_format': 'png',
            'splits': {
                'train': len(os.listdir(self.output_dir / 'train' / 'images')),
                'val': len(os.listdir(self.output_dir / 'val' / 'images')),
                'test': len(os.listdir(self.output_dir / 'test' / 'images'))
            }
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("=" * 60)
        logger.info(f"Dataset preparation complete!")
        logger.info(f"Saved to: {self.output_dir}")
        logger.info(f"Metadata: {metadata}")
        logger.info("=" * 60)
        
        return True


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Prepare satellite building segmentation dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/satellite_dataset',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of dataset'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    preparer = DatasetPreparer(
        output_dir=args.output_dir,
        force_download=args.force_download
    )
    
    success = preparer.prepare_dataset()
    exit(0 if success else 1)
