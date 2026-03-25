"""Generate prediction visualizations for test samples

Run this after training:
    python scripts/visualize_predictions.py
"""

import os
import numpy as np
from pathlib import Path
import logging

import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import load_img, img_to_array

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """IoU and Dice score computation"""
    
    @staticmethod
    @keras.saving.register_keras_serializable()
    def dice_loss(y_true, y_pred, smooth=1):
        intersection = tf.reduce_sum(y_true * y_pred)
        total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return 1 - (2 * intersection + smooth) / (total + smooth)


def load_test_samples(dataset_dir, num_samples=6):
    """Load test images and masks"""
    test_dir = Path(dataset_dir) / 'test'
    image_dir = test_dir / 'images'
    mask_dir = test_dir / 'masks'
    
    image_files = sorted(image_dir.glob('*.jpg'))[:num_samples]
    
    images = []
    masks = []
    filenames = []
    
    for img_file in image_files:
        # Load image
        image = load_img(str(img_file), target_size=(256, 256))
        image_array = img_to_array(image) / 255.0
        
        # Load mask
        mask_file = mask_dir / img_file.name.replace('.jpg', '.png')
        mask = load_img(str(mask_file), color_mode='grayscale', target_size=(256, 256))
        mask_array = img_to_array(mask) / 255.0
        
        images.append(image_array)
        masks.append(mask_array)
        filenames.append(img_file.stem)
    
    return np.array(images), np.array(masks), filenames


def visualize_predictions(model_path, dataset_dir, output_dir='screenshots', num_samples=6):
    """Generate and save prediction visualizations"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    if model is None:
        raise RuntimeError(f"Failed to load model from {model_path}")
    
    # Load test samples
    logger.info(f"Loading {num_samples} test samples...")
    X_test, y_test, filenames = load_test_samples(dataset_dir, num_samples)
    
    # Generate predictions
    logger.info("Generating predictions...")
    y_pred = model.predict(X_test, verbose='auto')
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 2 rows per sample: row 1 = original + true mask, row 2 = predicted + difference
    fig, axes = plt.subplots(num_samples, 3, figsize=(14, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(X_test[i])
        axes[i, 0].set_title(f'Original Image', fontweight='bold')
        axes[i, 0].axis('off')
        
        # True mask
        axes[i, 1].imshow(y_test[i], cmap='gray')
        axes[i, 1].set_title(f'True Mask', fontweight='bold')
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(y_pred[i], cmap='gray')
        axes[i, 2].set_title(f'Predicted Mask', fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    output_path = Path(output_dir) / f'predictions_grid.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()
    
    # Save individual predictions
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original
        axes[0].imshow(X_test[i])
        axes[0].set_title('Original', fontweight='bold')
        axes[0].axis('off')
        
        # True mask
        axes[1].imshow(y_test[i], cmap='gray')
        axes[1].set_title('True Mask', fontweight='bold')
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(y_pred[i], cmap='gray')
        axes[2].set_title('Predicted Mask', fontweight='bold')
        axes[2].axis('off')
        
        # Overlay (predicted on original)
        overlay = X_test[i].copy()
        overlay[:, :, 0] = np.where(y_pred[i, :, :, 0] > 0.5, 255, overlay[:, :, 0])
        axes[3].imshow(overlay / 255.0)
        axes[3].set_title('Predicted Overlay', fontweight='bold')
        axes[3].axis('off')
        
        plt.tight_layout()
        output_path = Path(output_dir) / f'prediction_{filenames[i]}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_path}")
        plt.close()
    
    logger.info("\nPrediction visualizations complete!")


def main():
    model_path = Path('models/segmentation_model.keras')
    dataset_dir = 'data/satellite_dataset'
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Run: python scripts/train_and_evaluate.py")
        return False
    
    visualize_predictions(
        model_path=str(model_path),
        dataset_dir=dataset_dir,
        output_dir='screenshots',
        num_samples=6
    )
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
