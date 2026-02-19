
# Imports
import pandas as pd
import numpy as np
from datasets import load_dataset
import chess
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, jaccard_score, classification_report,
    multilabel_confusion_matrix, confusion_matrix
)
import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Data Loading

def load_lichess_puzzles(sample_size=20000):
    print("="*80)
    print("CHESS TACTICAL MOTIF CLASSIFICATION - RESNET CNN")
    print("Assignment 6 - ResNet Architecture with Data Augmentation")
    print("="*80)
    
    print("\nLoading Lichess dataset...")
    dataset = load_dataset("lichess/chess-puzzles", split="train")
    df = pd.DataFrame(dataset)
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Dataset loaded: {len(df)} puzzles")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\n--- Sample Data ---")
    print(df[['Rating', 'Themes', 'FEN']].head())
    
    return df

# Feature Engineering - CNN Board Representation

def fen_to_board_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    
    piece_idx = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = square // 8
            file = square % 8
            
            channel = piece_idx[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            
            tensor[rank, file, channel] = 1.0
    
    return tensor

def prepare_data(df, top_n_motifs=10):
    print("\n=== Preparing Data for CNN ===")
    
    # Get top N motifs
    theme_counter = Counter()
    for themes in df['Themes']:
        if isinstance(themes, list):
            theme_counter.update(themes)
        elif isinstance(themes, str):
            theme_counter.update(themes.split())
    
    top_motifs = [motif for motif, _ in theme_counter.most_common(top_n_motifs)]
    print(f"Top {top_n_motifs} motifs: {top_motifs}")
    
    X = []
    y = []
    
    for idx, row in df.iterrows():
        try:
            # Convert FEN to board tensor
            fen = row['FEN']
            board_tensor = fen_to_board_tensor(fen)
            
            # Create multi-label target
            if isinstance(row['Themes'], list):
                themes = row['Themes']
            elif isinstance(row['Themes'], str):
                themes = row['Themes'].split()
            else:
                themes = []
            
            label = [1 if motif in themes else 0 for motif in top_motifs]
            
            X.append(board_tensor)
            y.append(label)
            
            if (len(X)) % 2000 == 0:
                print(f"Processed {len(X)} positions...")
        
        except Exception as e:
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nInput tensor shape: {X.shape}")
    print(f"Label matrix shape: {y.shape}")
    print(f"Labels per sample (avg): {y.sum(axis=1).mean():.2f}")
    
    return X, y, top_motifs

# Data Augmentation

def augment_board(board_tensor, label):
    # Random horizontal flip (50% probability)
    if tf.random.uniform(()) > 0.5:
        board_tensor = tf.image.flip_left_right(board_tensor)
    
    # Random vertical flip (50% probability)
    if tf.random.uniform(()) > 0.5:
        board_tensor = tf.image.flip_up_down(board_tensor)
    
    # Random 90° rotation (k=0, 1, 2, 3 for 0°, 90°, 180°, 270°)
    k = tf.random.uniform((), maxval=4, dtype=tf.int32)
    board_tensor = tf.image.rot90(board_tensor, k)
    
    return board_tensor, label

def create_augmented_dataset(X, y, batch_size=64, shuffle_buffer=10000):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(augment_board, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ResNet Model Architecture

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    
    # First conv block
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Second conv block
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection: match dimensions if needed
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add residual and apply activation
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

def build_resnet_cnn(input_shape=(8, 8, 12), num_classes=10):
    print("\n=== Building ResNet CNN Model ===")
    print("Architecture:")
    print("  Input: 8x8x12 board representation")
    print("  Initial Conv2D(32) -> BatchNorm -> ReLU")
    print("  2x ResidualBlock(32)")
    print("  MaxPooling2D(2x2)")
    print("  2x ResidualBlock(64)")
    print("  GlobalAveragePooling2D")
    print("  Dense(128) -> Dropout(0.5) -> Dense(10, sigmoid)")
    print("\nRationale:")
    print("  - Residual connections enable training deeper networks")
    print("  - Skip connections help gradient flow during backpropagation")
    print("  - GlobalAveragePooling reduces parameters vs Flatten")
    print("  - Multiple residual blocks capture hierarchical patterns")
    print("  - Sigmoid for independent multi-label classification")
    
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution block
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # First set of residual blocks (32 filters)
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    
    # Downsample
    x = layers.MaxPooling2D(2)(x)
    
    # Second set of residual blocks (64 filters)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    # Global pooling instead of flatten (reduces parameters)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense classification head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer: sigmoid for multi-label classification
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model

# Learning Rate Scheduling

def get_lr_schedule(initial_lr=0.001, decay_steps=1000, decay_rate=0.9):
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )
    return lr_schedule

def compile_model(model, lr_schedule):
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Note: BinaryAccuracy is per-label accuracy, not sample-wise exact match
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    print(f"\n{model.summary()}")
    
    return model

# Model Training

def train_model(model, train_dataset, X_val, y_val, epochs=30):
    print("\n=== Training ResNet CNN Model ===")
    print(f"Using data augmentation: rotation (0°/90°/180°/270°) + flipping")
    print(f"Validation samples: {len(X_val)}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: Exponential decay schedule")
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Additional callback to track learning rate
    class LRLogger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = self.model.optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                current_lr = float(lr.numpy())
            else:
                current_lr = float(keras.backend.get_value(lr))
            print(f"Epoch {epoch+1} - Learning rate: {current_lr:.6f}")
    
    lr_logger = LRLogger()
    
    print("\nTraining started...")
    history = model.fit(
        train_dataset,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[early_stopping, lr_logger],
        verbose=1
    )
    
    print("\nTraining complete!")
    return history

# Model Evaluation

def evaluate_model(model, X_test, y_test, motif_names, threshold=0.5):
    print("\n=== Model Evaluation ===")
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Overall metrics
    print("\n--- Overall Metrics ---")
    hamming = hamming_loss(y_test, y_pred)
    print(f"Hamming Loss: {hamming:.4f} (lower is better)")
    
    exact_match = accuracy_score(y_test, y_pred)
    print(f"Exact Match Ratio: {exact_match:.4f}")
    
    jaccard = jaccard_score(y_test, y_pred, average='samples', zero_division=0)
    print(f"Jaccard Score: {jaccard:.4f}")
    
    # Macro/Micro averaged metrics
    print("\n--- Averaged Metrics ---")
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    
    print(f"Macro-averaged Precision: {precision_macro:.4f}")
    print(f"Macro-averaged Recall: {recall_macro:.4f}")
    print(f"Macro-averaged F1-Score: {f1_macro:.4f}")
    print()
    print(f"Micro-averaged Precision: {precision_micro:.4f}")
    print(f"Micro-averaged Recall: {recall_micro:.4f}")
    print(f"Micro-averaged F1-Score: {f1_micro:.4f}")
    
    print("\n--- Per-Motif Metrics ---")
    per_motif_metrics = []
    for i, motif in enumerate(motif_names):
        motif_precision = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
        motif_recall = recall_score(y_test[:, i], y_pred[:, i], zero_division=0)
        motif_f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
        support = y_test[:, i].sum()
        
        per_motif_metrics.append({
            'motif': motif,
            'precision': motif_precision,
            'recall': motif_recall,
            'f1': motif_f1,
            'support': int(support)
        })
        
        print(f"{motif:20s} - Precision: {motif_precision:.3f}, Recall: {motif_recall:.3f}, "
              f"F1: {motif_f1:.3f}, Support: {int(support)}")
    
    return {
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'hamming_loss': hamming,
        'exact_match': exact_match,
        'jaccard': jaccard,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'per_motif_metrics': per_motif_metrics
    }

# Visualizations

def plot_training_history(history):
    print("\n=== Creating Training Visualizations ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#e74c3c')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#3498db')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Model Loss (Binary Crossentropy)', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Binary Accuracy curves (per-label, not sample-wise exact match)
    axes[0, 1].plot(history.history['binary_accuracy'], label='Training Binary Acc', linewidth=2, color='#e74c3c')
    axes[0, 1].plot(history.history['val_binary_accuracy'], label='Validation Binary Acc', linewidth=2, color='#3498db')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Binary Accuracy', fontsize=12)
    axes[0, 1].set_title('Binary Accuracy (Per-Label)', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision curves
    axes[1, 0].plot(history.history['precision'], label='Training Precision', linewidth=2, color='#e74c3c')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2, color='#3498db')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Precision', fontsize=12)
    axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall curves
    axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2, color='#e74c3c')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2, color='#3498db')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Recall', fontsize=12)
    axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('ResNet CNN Training History with Data Augmentation', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('Assignment6/training_history.png', dpi=300, bbox_inches='tight')
    print("Saved: Assignment6/training_history.png")

def plot_confusion_matrices(y_test, y_pred, motif_names):
    print("Creating per-motif confusion matrices...")
    
    n_motifs = len(motif_names)
    rows = 2
    cols = 5
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    axes = axes.ravel()
    
    for i, motif in enumerate(motif_names):
        # Use sklearn confusion_matrix
        cm = confusion_matrix(y_test[:, i], y_pred[:, i], labels=[0, 1])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                   ax=axes[i], cbar=False)
        axes[i].set_title(f'{motif}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Predicted', fontsize=9)
        axes[i].set_ylabel('True', fontsize=9)
    
    plt.suptitle('Confusion Matrices per Tactical Motif', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Assignment6/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Saved: Assignment6/confusion_matrices.png")

def plot_multilabel_confusion_aggregate(y_test, y_pred, motif_names):
    print("Creating multi-label confusion matrix aggregate...")
    
    # Get multi-label confusion matrix from sklearn
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    
    # Extract TP, FP, TN, FN for each motif
    tn = mcm[:, 0, 0]
    fp = mcm[:, 0, 1]
    fn = mcm[:, 1, 0]
    tp = mcm[:, 1, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Stacked bar chart of TP, FP, FN per motif
    x = np.arange(len(motif_names))
    
    axes[0].bar(x, tp, label='True Positive', color='#2ecc71')
    axes[0].bar(x, fp, bottom=tp, label='False Positive', color='#e74c3c')
    axes[0].bar(x, fn, bottom=tp+fp, label='False Negative', color='#f39c12')
    
    axes[0].set_xlabel('Tactical Motif', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Prediction Breakdown per Motif', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(motif_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Precision and Recall per motif
    precision_per_motif = tp / (tp + fp + 1e-10)
    recall_per_motif = tp / (tp + fn + 1e-10)
    
    axes[1].plot(x, precision_per_motif, 'o-', label='Precision', linewidth=2, markersize=8, color='#3498db')
    axes[1].plot(x, recall_per_motif, 's-', label='Recall', linewidth=2, markersize=8, color='#e74c3c')
    axes[1].set_xlabel('Tactical Motif', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Precision & Recall per Motif', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(motif_names, rotation=45, ha='right')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Assignment6/multilabel_confusion_aggregate.png', dpi=300, bbox_inches='tight')
    print("Saved: Assignment6/multilabel_confusion_aggregate.png")

def plot_f1_scores(metrics, motif_names):
    print("Creating F1 score visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Per-motif F1 scores (horizontal bar chart)
    f1_scores = [m['f1'] for m in metrics['per_motif_metrics']]
    colors = plt.cm.RdYlGn([f1/1.0 for f1 in f1_scores])  # Color by performance
    
    y_pos = np.arange(len(motif_names))
    bars = axes[0].barh(y_pos, f1_scores, color=colors)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(motif_names)
    axes[0].set_xlabel('F1-Score', fontsize=12)
    axes[0].set_title('F1-Score per Tactical Motif', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    for i, (score, bar) in enumerate(zip(f1_scores, bars)):
        axes[0].text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=9)
    
    # Overall metrics comparison
    metric_names = ['Exact\nMatch', 'Jaccard', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1\n(Macro)', 'Hamming\nLoss']
    metric_values = [
        metrics['exact_match'],
        metrics['jaccard'],
        metrics['precision_macro'],
        metrics['recall_macro'],
        metrics['f1_macro'],
        1 - metrics['hamming_loss']  # Invert so higher is better
    ]
    
    colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12', '#1abc9c']
    bars = axes[1].bar(metric_names, metric_values, color=colors)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Overall Model Performance Metrics', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('ResNet CNN Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Assignment6/performance_metrics.png', dpi=300, bbox_inches='tight')
    print("Saved: Assignment6/performance_metrics.png")

def plot_augmentation_examples(X_sample):
    print("Creating augmentation visualization...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Take a sample board
    sample_board = X_sample[0]
    
    # Combined piece visualization (all channels summed)
    def visualize_board(board_tensor):
        # Sum white and black pieces separately
        white_pieces = board_tensor[:, :, :6].sum(axis=-1)
        black_pieces = board_tensor[:, :, 6:].sum(axis=-1)
        # Create visualization: white=positive, black=negative
        combined = white_pieces - black_pieces
        return combined
    
    augmentations = [
        ('Original', sample_board),
        ('Horizontal Flip', tf.image.flip_left_right(sample_board).numpy()),
        ('Vertical Flip', tf.image.flip_up_down(sample_board).numpy()),
        ('Rotate 90°', tf.image.rot90(sample_board, k=1).numpy()),
        ('Rotate 180°', tf.image.rot90(sample_board, k=2).numpy()),
        ('Rotate 270°', tf.image.rot90(sample_board, k=3).numpy()),
        ('H-Flip + Rot90', tf.image.rot90(tf.image.flip_left_right(sample_board), k=1).numpy()),
        ('V-Flip + Rot90', tf.image.rot90(tf.image.flip_up_down(sample_board), k=1).numpy()),
    ]
    
    for ax, (title, board) in zip(axes.ravel(), augmentations):
        combined = visualize_board(board)
        im = ax.imshow(combined, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid
        for i in range(9):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
            ax.axvline(i - 0.5, color='black', linewidth=0.5)
    
    plt.suptitle('Data Augmentation Examples\n(Blue=White pieces, Red=Black pieces)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Assignment6/augmentation_examples.png', dpi=300, bbox_inches='tight')
    print("Saved: Assignment6/augmentation_examples.png")

# Main Execution

def main():
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load data
    df = load_lichess_puzzles(sample_size=20000)
    
    # Prepare data
    X, y, motif_names = prepare_data(df, top_n_motifs=10)
    
    # Train/validation/test split
    print("\n=== Data Splitting ===")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 = 0.15
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create augmented dataset
    print("\n=== Creating Augmented Dataset ===")
    print("Augmentation strategy:")
    print("  - Random horizontal flip (50% probability)")
    print("  - Random vertical flip (50% probability)")
    print("  - Random rotation (0°, 90°, 180°, or 270°)")
    
    batch_size = 64
    train_dataset = create_augmented_dataset(X_train, y_train, batch_size=batch_size)
    
    # Visualize augmentation examples
    plot_augmentation_examples(X_train)
    
    # Build model
    model = build_resnet_cnn(input_shape=(8, 8, 12), num_classes=len(motif_names))
    
    # Configure learning rate schedule
    print("\n=== Learning Rate Schedule ===")
    initial_lr = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    print(f"Initial learning rate: {initial_lr}")
    print(f"Decay steps: {decay_steps}")
    print(f"Decay rate: {decay_rate}")
    
    lr_schedule = get_lr_schedule(initial_lr, decay_steps, decay_rate)
    model = compile_model(model, lr_schedule)
    
    # Train model
    history = train_model(model, train_dataset, X_val, y_val, epochs=30)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, motif_names)
    
    # Create visualizations
    plot_training_history(history)
    plot_confusion_matrices(y_test, metrics['y_pred'], motif_names)
    plot_multilabel_confusion_aggregate(y_test, metrics['y_pred'], motif_names)
    plot_f1_scores(metrics, motif_names)
    
    # Save model
    model.save('Assignment6/chess_resnet_model.keras')
    print("\nModel saved: chess_resnet_model.keras")
    
    # Final summary
    print("\n" + "="*80)
    print("ASSIGNMENT 6 - ANALYSIS COMPLETE")
    print("="*80)
    
    print("\n--- Model Architecture Summary ---")
    print("- ResNet-like CNN with residual blocks")
    print("- Skip connections for better gradient flow")
    print("- GlobalAveragePooling instead of Flatten")
    print("- Total residual blocks: 4 (2x32 filters + 2x64 filters)")
    
    print("\n--- Data Augmentation Summary ---")
    print("- Random horizontal flip")
    print("- Random vertical flip")
    print("- Random 90°/180°/270° rotation")
    print("- Augmentation respects chess board symmetry")
    
    print("\n--- Training Configuration ---")
    print(f"- Batch size: {batch_size}")
    print(f"- Initial learning rate: {initial_lr}")
    print(f"- LR schedule: Exponential decay (rate={decay_rate}, steps={decay_steps})")
    print(f"- Early stopping: patience=5, restore_best_weights=True")
    
    print("\n--- Key Performance Metrics ---")
    print(f"1. Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"2. Exact Match Ratio: {metrics['exact_match']:.4f}")
    print(f"3. Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"4. Micro F1-Score: {metrics['f1_micro']:.4f}")
    print(f"5. Jaccard Score: {metrics['jaccard']:.4f}")
    
    print("\n--- Per-Motif F1 Scores ---")
    for m in metrics['per_motif_metrics']:
        print(f"  {m['motif']:20s}: {m['f1']:.3f}")
    
    print("\n--- Generated Files ---")
    print("  - Assignment6/augmentation_examples.png")
    print("  - Assignment6/training_history.png")
    print("  - Assignment6/confusion_matrices.png")
    print("  - Assignment6/multilabel_confusion_aggregate.png")
    print("  - Assignment6/performance_metrics.png")
    print("  - Assignment6/chess_resnet_model.keras")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
