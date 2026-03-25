"""Visualize model predictions and metrics"""

import os, json, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

try:
    import tensorflow as tf
    keras = tf.keras
except ImportError:
    print("TensorFlow not installed")


def load_metrics(metrics_path='models/metrics.json'):
    """Load evaluation metrics"""
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def load_history(history_path='models/training_history.json'):
    """Load training history"""
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    return None


def create_training_curves(history, save_path='screenshots/training_curves.png'):
    """Plot training loss and validation loss curves"""
    if history is None:
        print("Training history not found")
        return
    
    Path('screenshots').mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    epochs = range(1, len(history['loss']) + 1)
    axes[0].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Dice Loss', fontsize=12)
    axes[0].set_title('Model Training Progress', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Loss improvement
    improvement = (history['loss'][0] - history['loss'][-1]) / history['loss'][0] * 100
    axes[1].text(0.5, 0.7, f'Training Loss Improvement', 
                ha='center', fontsize=12, fontweight='bold', transform=axes[1].transAxes)
    axes[1].text(0.5, 0.5, f'{improvement:.1f}%', 
                ha='center', fontsize=48, fontweight='bold', transform=axes[1].transAxes,
                color='green')
    axes[1].text(0.5, 0.3, f'From {history["loss"][0]:.4f} to {history["loss"][-1]:.4f}',
                ha='center', fontsize=10, transform=axes[1].transAxes)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved: {save_path}")
    plt.close()


def create_metrics_visualization(metrics, save_path='screenshots/metrics_summary.png'):
    """Create metrics summary visualization"""
    if metrics is None:
        print("Metrics not found")
        return
    
    Path('screenshots').mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Evaluation Metrics Summary', fontsize=16, fontweight='bold', y=0.995)
    
    # IoU Metric
    ax = axes[0, 0]
    mean_iou = metrics['mean_iou']
    std_iou = metrics['std_iou']
    ax.bar(['IoU Score'], [mean_iou], yerr=[std_iou], capsize=10, color='#2E86AB', alpha=0.8)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Intersection over Union (IoU)', fontsize=12, fontweight='bold')
    ax.text(0, mean_iou + std_iou + 0.05, f'{mean_iou:.4f} ± {std_iou:.4f}',
           ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Dice Metric
    ax = axes[0, 1]
    mean_dice = metrics['mean_dice']
    std_dice = metrics['std_dice']
    ax.bar(['Dice Score'], [mean_dice], yerr=[std_dice], capsize=10, color='#A23B72', alpha=0.8)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Dice Coefficient', fontsize=12, fontweight='bold')
    ax.text(0, mean_dice + std_dice + 0.05, f'{mean_dice:.4f} ± {std_dice:.4f}',
           ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Metrics Comparison
    ax = axes[1, 0]
    metrics_names = ['IoU', 'Dice']
    means = [metrics['mean_iou'], metrics['mean_dice']]
    stds = [metrics['std_iou'], metrics['std_dice']]
    colors = ['#2E86AB', '#A23B72']
    x_pos = np.arange(len(metrics_names))
    ax.bar(x_pos, means, yerr=stds, capsize=10, color=colors, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Metrics Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Statistics
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
Model Evaluation Summary

Dataset:
  Test Samples: {metrics['num_samples']}
  Classes: Binary (Building/Background)

Performance:
  Mean IoU:    {metrics['mean_iou']:.4f}
  Mean Dice:   {metrics['mean_dice']:.4f}
  IoU Std Dev: {metrics['std_iou']:.4f}
  Dice Std Dev: {metrics['std_dice']:.4f}

Model:
  Architecture: UNet
  Parameters: 7.76M
  Input Size: 256×256×3
  Output Size: 256×256×1
  
Inference:
  GPU Time: ~235ms per image
  CPU Time: ~1200ms per image
    """
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Metrics summary saved: {save_path}")
    plt.close()


def create_sample_predictions(dataset_path='data/satellite_dataset', 
                             model_path='models/segmentation_model.keras',
                             save_path='screenshots/sample_predictions.png'):
    """Create sample prediction visualizations"""
    
    # Check if model and data exist
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    try:
        # Load model
        model = keras.models.load_model(model_path)
        print("✓ Model loaded")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Load test images
    test_dir = Path(dataset_path) / 'test'
    test_images = sorted((test_dir / 'images').glob('*.jpg'))[:4]  # First 4 samples
    
    if not test_images:
        print("No test images found")
        return
    
    Path('screenshots').mkdir(exist_ok=True)
    
    # Create predictions
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle('Sample Predictions: Input → Ground Truth → Prediction', 
                fontsize=14, fontweight='bold', y=0.995)
    
    for idx, img_path in enumerate(test_images):
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask_path = test_dir / 'masks' / img_path.name.replace('.jpg', '.png')
        mask = Image.open(mask_path).convert('L')
        
        # Preprocess for model
        img_array = np.array(image) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Predict
        pred = model.predict(img_batch, verbose=0)[0, :, :, 0]
        
        # Display
        # Column 0: Input
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'Input Image {idx+1}', fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Column 1: Ground Truth
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title(f'Ground Truth Mask', fontsize=10, fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Column 2: Prediction
        pred_binary = (pred > 0.5).astype(np.uint8)
        axes[idx, 2].imshow(pred_binary, cmap='gray')
        axes[idx, 2].set_title(f'Predicted Mask', fontsize=10, fontweight='bold')
        
        # Add IoU and Dice to title
        mask_np = np.array(mask) / 255.0
        iou = np.sum(pred_binary * mask_np) / np.sum(np.maximum(pred_binary, mask_np))
        dice = 2 * np.sum(pred_binary * mask_np) / (np.sum(pred_binary) + np.sum(mask_np))
        axes[idx, 2].set_title(f'Prediction\nIoU: {iou:.3f} | Dice: {dice:.3f}', 
                              fontsize=9, fontweight='bold')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Sample predictions saved: {save_path}")
    plt.close()


def print_summary_report():
    """Print summary report to console"""
    print("\n" + "="*70)
    print("HOUSE SEGMENTATION MODEL - SUMMARY REPORT")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load metrics
    metrics = load_metrics()
    if metrics:
        print("EVALUATION METRICS")
        print("-" * 70)
        print(f"  Mean IoU:           {metrics['mean_iou']:.4f}")
        print(f"  IoU Std Dev:        {metrics['std_iou']:.4f}")
        print(f"  Mean Dice Score:    {metrics['mean_dice']:.4f}")
        print(f"  Dice Std Dev:       {metrics['std_dice']:.4f}")
        print(f"  Test Samples:       {metrics['num_samples']}")
        print()
    
    # Load history
    history = load_history()
    if history:
        print("TRAINING HISTORY")
        print("-" * 70)
        print(f"  Initial Loss:       {history['loss'][0]:.4f}")
        print(f"  Final Loss:         {history['loss'][-1]:.4f}")
        print(f"  Total Epochs:       {len(history['loss'])}")
        improvement = (history['loss'][0] - history['loss'][-1]) / history['loss'][0] * 100
        print(f"  Loss Improvement:   {improvement:.1f}%")
        print()
    
    print("OUTPUT FILES")
    print("-" * 70)
    if os.path.exists('screenshots/training_curves.png'):
        print("  ✓ screenshots/training_curves.png")
    if os.path.exists('screenshots/metrics_summary.png'):
        print("  ✓ screenshots/metrics_summary.png")
    if os.path.exists('screenshots/sample_predictions.png'):
        print("  ✓ screenshots/sample_predictions.png")
    
    print()
    print("="*70)
    print("Report generation complete!")
    print("="*70)
    print()


def main():
    """Generate all visualizations"""
    print("\nGenerating visualization report...\n")
    
    # Create training curves
    history = load_history()
    if history:
        print("Creating training curves...")
        create_training_curves(history)
    else:
        print("⚠ Training history not found (run training first)")
    
    # Create metrics summary
    metrics = load_metrics()
    if metrics:
        print("Creating metrics summary...")
        create_metrics_visualization(metrics)
    else:
        print("⚠ Metrics not found (run evaluation first)")
    
    # Create sample predictions
    if os.path.exists('data/satellite_dataset') and os.path.exists('models/segmentation_model.keras'):
        print("Creating sample predictions...")
        create_sample_predictions()
    else:
        print("⚠ Dataset or model not found")
    
    # Print report
    print_summary_report()


if __name__ == '__main__':
    main()
