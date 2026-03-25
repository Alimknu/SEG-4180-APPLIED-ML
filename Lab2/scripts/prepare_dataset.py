"""Download and prepare satellite building segmentation dataset

Using Week 7 pixel mask generation code to create labeled dataset
from official satellite-building-segmentation dataset or synthetic fallback
"""

import os, json, numpy as np
from PIL import Image
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    def __init__(self, output_dir='data/satellite_dataset', use_synthetic=False):
        self.output_dir = Path(output_dir)
        self.use_synthetic = use_synthetic
        self.dataset = None
        # Create output directories
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'masks']:
                (self.output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    def download_satellite_dataset(self):
        """Download satellite building segmentation dataset"""
        logger.info("Downloading satellite building segmentation dataset...")
        try:
            from datasets import load_dataset
            # Load from Hugging Face Hub
            self.dataset = load_dataset(
                "keremberke/satellite-building-segmentation",
                name="full"
            )
            logger.info(f"Dataset loaded with splits: {list(self.dataset.keys())}")
            logger.info(f"Training samples: {len(self.dataset['train'])}")
            return True
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return False
    
    def make_mask(self, bbox, image_width, image_height):
        """Create binary mask from bounding box (Week 7 pixel mask generation code)
        
        From Week 7 code:
            def make_mask(labelled_bbox, image):
              x_min_ones, y_min_ones, width_ones, height_ones = labelled_bbox
              x_min_ones, y_min_ones, width_ones, height_ones = int(...), int(...), int(...), int(...)
              mask_instance = np.zeros((image.width,image.height))
              last_x = x_min_ones+width_ones
              last_y = y_min_ones+height_ones
              mask_instance[x_min_ones:last_x, y_min_ones:last_y] = np.ones(...)
              return mask_instance.T
        """
        x_min, y_min, width, height = [int(v) for v in bbox]
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        x_max = min(x_min + width, image_width)
        y_max = min(y_min + height, image_height)
        mask[y_min:y_max, x_min:x_max] = 1
        return mask
    
    def create_composite_mask(self, example):
        """Create composite mask from all building bboxes"""
        image = example['image']
        width, height = image.size
        composite_mask = np.zeros((height, width), dtype=np.uint8)
        
        if 'objects' in example and 'bbox' in example['objects']:
            for bbox in example['objects']['bbox']:
                mask = self.make_mask(bbox, width, height)
                composite_mask = np.maximum(composite_mask, mask)
        
        return composite_mask
    
    def prepare_real_dataset(self, num_samples=None):
        """Prepare real satellite dataset from HuggingFace"""
        if not self.download_satellite_dataset():
            return False
        
        train_data = self.dataset['train']
        num_samples = num_samples or len(train_data)
        
        # Create train/val/test split
        indices = list(range(min(num_samples, len(train_data))))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        
        splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        
        for split_type, idx_list in splits.items():
            logger.info(f"Processing {split_type} split ({len(idx_list)} samples)...")
            
            for i, sample_idx in enumerate(tqdm(idx_list, desc=split_type)):
                try:
                    example = train_data[sample_idx]
                    image = example['image'].convert('RGB')
                    image = image.resize((256, 256), Image.BILINEAR)
                    
                    mask = self.create_composite_mask(example)
                    if mask.shape != (256, 256):
                        mask = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))
                    
                    # Skip if no buildings
                    if np.sum(mask) < 50:
                        continue
                    
                    # Save
                    img_path = self.output_dir / split_type / 'images' / f'{i:04d}.jpg'
                    mask_path = self.output_dir / split_type / 'masks' / f'{i:04d}.png'
                    
                    image.save(img_path, quality=95)
                    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                
                except Exception as e:
                    logger.warning(f"Skipped sample {sample_idx}: {e}")
        
        return True
    
    def generate_synthetic_image_and_mask(self, seed=None):
        """Generate synthetic satellite image with building mask"""
        if seed is not None:
            np.random.seed(seed)
        
        # Base image (aerial view-like)
        image = np.random.randint(40, 120, (256, 256, 3), dtype=np.uint8)
        
        # Add grass/street texture
        for _ in range(3):
            y, x = np.random.randint(0, 200), np.random.randint(0, 200)
            size = np.random.randint(30, 100)
            color = np.random.randint(20, 80, 3)
            y_end = min(y + size, 256)
            x_end = min(x + size, 256)
            image[y:y_end, x:x_end] = color
        
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Create buildings
        mask = np.zeros((256, 256), dtype=np.uint8)
        num_buildings = np.random.randint(4, 12)
        
        for _ in range(num_buildings):
            x = np.random.randint(10, 240)
            y = np.random.randint(10, 240)
            w = np.random.randint(12, 50)
            h = np.random.randint(12, 50)
            
            x_end = min(x + w, 256)
            y_end = min(y + h, 256)
            
            mask[y:y_end, x:x_end] = 1
            roof_color = np.random.randint(100, 200, 3)
            image[y:y_end, x:x_end] = roof_color
        
        return image, mask
    
    def prepare_synthetic_dataset(self, num_samples=300):
        """Generate synthetic satellite dataset"""
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        # Split: 70% train, 15% val, 15% test
        num_train = int(num_samples * 0.7)
        num_val = int(num_samples * 0.15)
        num_test = num_samples - num_train - num_val
        
        splits = {'train': num_train, 'val': num_val, 'test': num_test}
        
        global_idx = 0
        for split_type, count in splits.items():
            logger.info(f"Generating {split_type} split ({count} samples)...")
            
            for i in tqdm(range(count), desc=split_type):
                image, mask = self.generate_synthetic_image_and_mask(seed=global_idx)
                
                img_path = self.output_dir / split_type / 'images' / f'{i:04d}.jpg'
                mask_path = self.output_dir / split_type / 'masks' / f'{i:04d}.png'
                
                Image.fromarray(image).save(img_path, quality=95)
                Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                
                global_idx += 1
        
        return True
    
    def prepare_dataset(self, num_samples=None):
        """Prepare dataset (real or synthetic)"""
        logger.info("=" * 60)
        logger.info("Satellite Building Segmentation Dataset Preparation")
        logger.info("=" * 60)
        
        if self.use_synthetic:
            success = self.prepare_synthetic_dataset(num_samples or 300)
        else:
            try:
                success = self.prepare_real_dataset(num_samples)
            except Exception as e:
                logger.warning(f"Real dataset failed: {e}")
                logger.info("Falling back to synthetic dataset...")
                success = self.prepare_synthetic_dataset(num_samples or 300)
        
        if success:
            # Save metadata
            splits_info = {}
            for split in ['train', 'val', 'test']:
                img_count = len(list((self.output_dir / split / 'images').glob('*.jpg')))
                splits_info[split] = img_count
            
            metadata = {
                'dataset_name': 'Satellite Building Segmentation',
                'source': 'HuggingFace keremberke/satellite-building-segmentation (or synthetic)',
                'task': 'semantic_segmentation',
                'num_classes': 2,
                'class_names': ['background', 'building'],
                'image_size': [256, 256],
                'image_format': 'jpg',
                'mask_format': 'png',
                'splits': splits_info
            }
            
            metadata_path = self.output_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("=" * 60)
            logger.info(f"Dataset preparation complete!")
            logger.info(f"Saved to: {self.output_dir}")
            for split, count in splits_info.items():
                logger.info(f"  {split}: {count} samples")
            logger.info("=" * 60)
        
        return success


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Prepare satellite building segmentation dataset')
    parser.add_argument('--output-dir', type=str, default='data/satellite_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--synthetic', action='store_true', default=True,
                       help='Generate synthetic dataset (default)')
    parser.add_argument('--real', action='store_true', dest='use_real',
                       help='Use real dataset from HuggingFace')
    parser.add_argument('--num-samples', type=int, default=300,
                       help='Number of samples to generate')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    preparer = DatasetPreparer(
        output_dir=args.output_dir,
        use_synthetic=not args.use_real
    )
    
    success = preparer.prepare_dataset(args.num_samples)
    exit(0 if success else 1)
