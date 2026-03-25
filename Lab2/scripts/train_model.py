"""Train UNet segmentation model on satellite building dataset"""

import os, json, numpy as np
from pathlib import Path
from datetime import datetime
import logging, argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

from config import AppConfig, setup_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """IoU and Dice score computation"""
    
    @staticmethod
    def iou_score(y_true, y_pred, threshold=0.5):
        """Intersection over Union"""
        y_pred = (y_pred > threshold).astype(np.uint8)
        y_true = y_true.astype(np.uint8)
        intersection = np.sum(y_true * y_pred)
        union = np.sum(np.maximum(y_true, y_pred))
        return intersection / union if union > 0 else 1.0
    
    @staticmethod
    def dice_score(y_true, y_pred, threshold=0.5):
        """Dice Coefficient"""
        y_pred = (y_pred > threshold).astype(np.uint8)
        y_true = y_true.astype(np.uint8)
        intersection = np.sum(y_true * y_pred)
        total = np.sum(y_true) + np.sum(y_pred)
        return (2 * intersection / total) if total > 0 else 1.0
    
    @staticmethod
    def dice_loss(y_true, y_pred, smooth=1):
        intersection = tf.reduce_sum(y_true * y_pred)
        total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return 1 - (2 * intersection + smooth) / (total + smooth)


class UNetSegmentationModel:
    def __init__(self, input_size=(256, 256, 3), num_classes=1):
        self.input_size = input_size
        self.num_classes = num_classes
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
        outputs = layers.Conv2D(
            self.num_classes, (1, 1), activation='sigmoid'
        )(c7)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile(self, learning_rate=0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=SegmentationMetrics.dice_loss,
            metrics=['mse']
        )
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()


class SegmentationDataLoader:
    def __init__(self, dataset_dir, img_size=(256, 256)):
        self.dataset_dir = Path(dataset_dir)
        self.img_size = img_size
    
    def load_image_mask_pair(self, img_path, mask_path):
        """Load and preprocess image-mask pair"""
        # Load image
        image = load_img(img_path, target_size=self.img_size)
        image = img_to_array(image) / 255.0
        
        # Load mask
        mask = load_img(mask_path, color_mode='grayscale', target_size=self.img_size)
        mask = img_to_array(mask) / 255.0
        
        return image, mask
    
    def load_split(self, split='train'):
        """Load train/val/test split"""
        split_dir = self.dataset_dir / split
        if not split_dir.exists():
            raise ValueError(f"Dataset split {split} not found at {split_dir}")
        
        image_dir = split_dir / 'images'
        mask_dir = split_dir / 'masks'
        
        image_files = sorted(image_dir.glob('*.jpg'))
        logger.info(f"Loading {split} dataset ({len(image_files)} samples)...")
        
        images = []
        masks = []
        
        for img_file in tqdm(image_files, desc=f"Loading {split}"):
            mask_file = mask_dir / img_file.name.replace('.jpg', '.png')
            
            if not mask_file.exists():
                logger.warning(f"Mask not found for {img_file.name}, skipping")
                continue
            
            try:
                image, mask = self.load_image_mask_pair(str(img_file), str(mask_file))
                images.append(image)
                masks.append(mask)
            except Exception as e:
                logger.error(f"Error loading {img_file.name}: {e}")
        
        return np.array(images), np.array(masks)
    
    def get_data_generators(self, split='train', batch_size=32):
        """Get data generators with augmentation"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        return datagen


class SegmentationTrainer:
    def __init__(self, model, dataset_dir, output_dir='models'):
        """
        Initialize trainer
        
        Args:
            model: UNet model instance
            dataset_dir: Path to dataset
            output_dir: Directory for saving model and logs
        """
        self.model = model
        self.dataset_dir = dataset_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data_loader = SegmentationDataLoader(dataset_dir)
        self.history = None
        self.metrics_dict = {}
    
    def train(self, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            
        Returns:
            Training history
        """
        logger.info("Loading training dataset...")
        X_train, y_train = self.data_loader.load_split('train')
        logger.info(f"Loaded {len(X_train)} training samples")
        
        # Create validation split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=validation_split,
            random_state=42
        )
        
        logger.info(f"Training split: {len(X_train_split)}, "
                   f"Validation split: {len(X_val)}")
        
        # Callbacks
        checkpoint_path = self.output_dir / 'segmentation_model.keras'
        callbacks = [
            ModelCheckpoint(
                str(checkpoint_path),
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1
            ),
            TensorBoard(
                log_dir=str(self.output_dir / 'logs'),
                update_freq='epoch'
            )
        ]
        
        # Train
        logger.info("Starting training...")
        self.history = self.model.model.fit(
            X_train_split, y_train_split,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training complete!")
        return self.history
    
    def evaluate(self):
        """Evaluate model on test set"""
        logger.info("Loading test dataset...")
        X_test, y_test = self.data_loader.load_split('test')
        logger.info(f"Loaded {len(X_test)} test samples")
        
        logger.info("Evaluating model...")
        y_pred = self.model.model.predict(X_test, verbose=0)
        
        # Compute metrics
        ious = []
        dices = []
        
        for i in tqdm(range(len(X_test)), desc="Computing metrics"):
            iou = SegmentationMetrics.iou_score(y_test[i], y_pred[i])
            dice = SegmentationMetrics.dice_score(y_test[i], y_pred[i])
            ious.append(iou)
            dices.append(dice)
        
        self.metrics_dict = {
            'mean_iou': float(np.mean(ious)),
            'std_iou': float(np.std(ious)),
            'mean_dice': float(np.mean(dices)),
            'std_dice': float(np.std(dices)),
            'num_samples': len(X_test)
        }
        
        logger.info("=" * 60)
        logger.info("Evaluation Metrics:")
        logger.info(f"  Mean IoU:  {self.metrics_dict['mean_iou']:.4f} "
                   f"(±{self.metrics_dict['std_iou']:.4f})")
        logger.info(f"  Mean Dice: {self.metrics_dict['mean_dice']:.4f} "
                   f"(±{self.metrics_dict['std_dice']:.4f})")
        logger.info("=" * 60)
        
        return self.metrics_dict
    
    def save_metrics(self):
        """Save metrics to file"""
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_dict, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save history
        if self.history:
            history_path = self.output_dir / 'training_history.json'
            history_dict = {
                'loss': [float(v) for v in self.history.history['loss']],
                'val_loss': [float(v) for v in self.history.history['val_loss']],
            }
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
            logger.info(f"History saved to {history_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--dataset-dir', type=str, default='data/satellite_dataset',
                       help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate on test set after training')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Segmentation Model Training")
    logger.info("=" * 60)
    
    # Build model
    logger.info("Building UNet model...")
    unet = UNetSegmentationModel(input_size=(256, 256, 3), num_classes=1)
    unet.build()
    unet.compile(learning_rate=args.learning_rate)
    unet.summary()
    
    # Train
    trainer = SegmentationTrainer(
        unet,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir
    )
    
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate
    if args.evaluate:
        trainer.evaluate()
        trainer.save_metrics()
    
    logger.info("Training script completed!")
