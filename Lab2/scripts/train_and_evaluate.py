"""Train UNet model for satellite building segmentation

Step 3: Dataset Preparation
Step 4: Model Training with IoU/Dice metrics
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

import tensorflow as tf
import keras
from keras import layers, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """IoU and Dice score computation"""
    
    @staticmethod
    def iou_score(y_true, y_pred, threshold=0.5):
        y_pred = (y_pred > threshold).astype(np.uint8)
        y_true = y_true.astype(np.uint8)
        intersection = np.sum(y_true * y_pred)
        union = np.sum(np.maximum(y_true, y_pred))
        return intersection / union if union > 0 else 1.0
    
    @staticmethod
    def dice_score(y_true, y_pred, threshold=0.5):
        y_pred = (y_pred > threshold).astype(np.uint8)
        y_true = y_true.astype(np.uint8)
        intersection = np.sum(y_true * y_pred)
        total = np.sum(y_true) + np.sum(y_pred)
        return (2 * intersection / total) if total > 0 else 1.0
    
    @staticmethod
    @keras.saving.register_keras_serializable()
    def dice_loss(y_true, y_pred, smooth=1):
        intersection = tf.reduce_sum(y_true * y_pred)
        total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return 1 - (2 * intersection + smooth) / (total + smooth)


class UNet:
    def __init__(self, input_size=(256, 256, 3)):
        self.input_size = input_size
        self.model = None
    
    def build(self):
        """Build UNet architecture"""
        inputs = keras.Input(shape=self.input_size)
        
        # Encoder
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
        
        # Decoder
        u5 = layers.UpSampling2D((2, 2))(c4)
        u5 = layers.concatenate([u5, c3])
        c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
        c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
        
        u6 = layers.UpSampling2D((2, 2))(c5)
        u6 = layers.concatenate([u6, c2])
        c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
        
        u7 = layers.UpSampling2D((2, 2))(c6)
        u7 = layers.concatenate([u7, c1])
        c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
        
        # Output
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile(self, learning_rate=0.001):
        """Compile the model"""
        if self.model is None:
            raise RuntimeError("Model must be built before compilation. Call build() first.")
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=SegmentationMetrics.dice_loss,
            metrics=['mse']
        )
    
    def summary(self):
        if self.model:
            self.model.summary()


class DataLoader:
    def __init__(self, dataset_dir, img_size=(256, 256)):
        self.dataset_dir = Path(dataset_dir)
        self.img_size = img_size
    
    def load_image_mask_pair(self, img_path, mask_path):
        image = load_img(img_path, target_size=self.img_size)
        image = img_to_array(image) / 255.0
        
        mask = load_img(mask_path, color_mode='grayscale', target_size=self.img_size)
        mask = img_to_array(mask) / 255.0
        
        return image, mask
    
    def load_split(self, split='train'):
        split_dir = self.dataset_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split {split} not found at {split_dir}")
        
        image_dir = split_dir / 'images'
        mask_dir = split_dir / 'masks'
        
        image_files = sorted(image_dir.glob('*.jpg'))
        logger.info(f"Loading {split}: {len(image_files)} samples")
        
        images = []
        masks = []
        
        for img_file in tqdm(image_files, desc=f"Loading {split}"):
            mask_file = mask_dir / img_file.name.replace('.jpg', '.png')
            
            if not mask_file.exists():
                logger.warning(f"Mask not found for {img_file.name}")
                continue
            
            try:
                image, mask = self.load_image_mask_pair(str(img_file), str(mask_file))
                images.append(image)
                masks.append(mask)
            except Exception as e:
                logger.error(f"Error loading {img_file.name}: {e}")
        
        return np.array(images), np.array(masks)


class ModelTrainer:
    def __init__(self, model, dataset_dir, output_dir='models'):
        self.model = model
        self.dataset_dir = dataset_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data_loader = DataLoader(dataset_dir)
        self.history = None
    
    def train(self, epochs=30, batch_size=16, validation_split=0.2):
        logger.info("\nLoading training data...")
        X_train, y_train = self.data_loader.load_split('train')
        logger.info(f"Training samples: {len(X_train)}")
        
        # Validation split
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42
        )
        
        logger.info(f"Train: {len(X_train_split)}, Val: {len(X_val)}")
        
        # Callbacks
        checkpoint_path = self.output_dir / 'segmentation_model.keras'
        callbacks = [
            ModelCheckpoint(str(checkpoint_path), save_best_only=True, 
                          monitor='val_loss', verbose=0),
            EarlyStopping(monitor='val_loss', patience=8, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=0),
        ]
        
        logger.info("\nStarting training...")
        self.history = self.model.model.fit(
            X_train_split, y_train_split,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"\nModel saved: {checkpoint_path}")
        return self.history
    
    def evaluate(self):
        logger.info("\nLoading test data...")
        X_test, y_test = self.data_loader.load_split('test')
        logger.info(f"Test samples: {len(X_test)}")
        
        logger.info("Evaluating model...")
        y_pred = self.model.model.predict(X_test, verbose=0)
        
        ious = []
        dices = []
        
        for i in tqdm(range(len(X_test))):
            iou = SegmentationMetrics.iou_score(y_test[i], y_pred[i])
            dice = SegmentationMetrics.dice_score(y_test[i], y_pred[i])
            ious.append(iou)
            dices.append(dice)
        
        results = {
            'mean_iou': float(np.mean(ious)),
            'std_iou': float(np.std(ious)),
            'mean_dice': float(np.mean(dices)),
            'std_dice': float(np.std(dices)),
            'num_samples': len(X_test)
        }
        
        logger.info("\n" + "="*50)
        logger.info("EVALUATION RESULTS")
        logger.info("="*50)
        logger.info(f"Mean IoU:  {results['mean_iou']:.4f} (±{results['std_iou']:.4f})")
        logger.info(f"Mean Dice: {results['mean_dice']:.4f} (±{results['std_dice']:.4f})")
        logger.info("="*50)
        
        return results


def main(args):
    logger.info("\n" + "="*50)
    logger.info("STEP 3: Dataset Preparation")
    logger.info("STEP 4: Model Training")
    logger.info("="*50)
    
    # Verify dataset
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {args.dataset_dir}")
        logger.info("Run: python scripts/prepare_dataset_real.py")
        return False
    
    # Build model
    logger.info("\nBuilding UNet model...")
    unet = UNet(input_size=(256, 256, 3))
    unet.build()
    unet.compile(learning_rate=0.001)
    unet.summary()
    
    # Train
    trainer = ModelTrainer(unet, dataset_dir=args.dataset_dir, output_dir=args.output_dir)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluate
    results = trainer.evaluate()
    
    # Save results
    results_path = Path(args.output_dir) / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved: {results_path}")
    
    # Generate visualizations
    _generate_visualizations(results)
    
    logger.info("="*50 + "\n")
    
    return True


def _generate_visualizations(results):
    """Generate PNG visualizations"""
    import matplotlib.pyplot as plt
    
    Path('screenshots').mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('UNet Segmentation - Evaluation Metrics', fontsize=14, fontweight='bold')
    
    # IoU
    ax = axes[0, 0]
    ax.bar(['IoU'], [results['mean_iou']], yerr=[results['std_iou']], 
           capsize=10, color='#2E86AB', alpha=0.8, width=0.5)
    ax.set_ylim([0, 1])
    ax.set_title('Intersection over Union', fontweight='bold')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Dice
    ax = axes[0, 1]
    ax.bar(['Dice'], [results['mean_dice']], yerr=[results['std_dice']], 
           capsize=10, color='#A23B72', alpha=0.8, width=0.5)
    ax.set_ylim([0, 1])
    ax.set_title('Dice Coefficient', fontweight='bold')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Comparison
    ax = axes[1, 0]
    metrics_names = ['IoU', 'Dice']
    means = [results['mean_iou'], results['mean_dice']]
    stds = [results['std_iou'], results['std_dice']]
    ax.bar(metrics_names, means, yerr=stds, capsize=10, 
           color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1])
    ax.set_title('Metrics Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""Model Summary

Test Samples: {results['num_samples']}

IoU:
  Mean: {results['mean_iou']:.4f}
  Std: {results['std_iou']:.4f}

Dice:
  Mean: {results['mean_dice']:.4f}
  Std: {results['std_dice']:.4f}

Architecture: UNet
Parameters: 7.78M
Input: 256x256x3
Output: 256x256x1"""
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('screenshots/metrics_summary.png', dpi=150, bbox_inches='tight')
    logger.info("✓ Visualization saved: screenshots/metrics_summary.png")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet segmentation model')
    parser.add_argument('--dataset-dir', type=str, default='data/satellite_dataset',
                       help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    success = main(args)
    exit(0 if success else 1)
