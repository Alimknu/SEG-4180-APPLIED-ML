
# Imports
import pandas as pd
import numpy as np
from datasets import load_dataset
import chess
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, jaccard_score, classification_report,
    multilabel_confusion_matrix
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
    print("CHESS TACTICAL MOTIF CLASSIFICATION - DEEP LEARNING")
    print("Assignment 5 - CNN for Chess Pattern Recognition")
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
    """
    Convert FEN to 8x8x12 tensor for CNN input.
    
    Channels (12):
    - 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    - 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    
    Each channel is an 8x8 binary matrix indicating piece presence.
    """
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

# CNN Model Architecture

def build_cnn_model(input_shape, num_classes):
    print("\n=== Building CNN Model ===")
    print("Architecture:")
    print("  Input: 8x8x12 board representation")
    print("  Conv2D(64) -> ReLU -> BatchNorm")
    print("  Conv2D(128) -> ReLU -> BatchNorm -> MaxPooling")
    print("  Conv2D(256) -> ReLU -> BatchNorm")
    print("  Flatten -> Dense(256) -> Dropout -> Dense(128) -> Dropout")
    print("  Output: Dense(10) with Sigmoid (multi-label)")
    print("\nRationale:")
    print("  - CNNs learn spatial patterns (piece relationships, tactical patterns)")
    print("  - Multiple conv layers capture different abstraction levels")
    print("  - BatchNorm for stable training")
    print("  - Dropout prevents overfitting")
    print("  - Sigmoid for independent multi-label predictions")
    
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        
        # Second Conv Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third Conv Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output Layer
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    print(f"\n{model.summary()}")
    
    return model

# Model Training

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    print("\n=== Training CNN Model ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    print("\nTraining started...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
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
    
    # Per-motif metrics
    print("\n--- Per-Motif Metrics ---")
    for i, motif in enumerate(motif_names):
        precision = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(y_test[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
        support = y_test[:, i].sum()
        
        print(f"{motif:20s} - Precision: {precision:.3f}, Recall: {recall:.3f}, "
              f"F1: {f1:.3f}, Support: {int(support)}")
    
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
        'f1_micro': f1_micro
    }

# Visualizations

def plot_training_history(history):
    print("\n=== Creating Training Visualizations ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Precision', fontsize=12)
    axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Recall', fontsize=12)
    axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Saved: training_history.png")

def plot_confusion_matrices(y_test, y_pred, motif_names):
    print("Creating confusion matrices...")
    
    n_motifs = len(motif_names)
    rows = 2
    cols = 5
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    axes = axes.ravel()
    
    for i, motif in enumerate(motif_names):
        cm = np.zeros((2, 2))
        for true, pred in zip(y_test[:, i], y_pred[:, i]):
            cm[int(true), int(pred)] += 1
        
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                   ax=axes[i], cbar=False)
        axes[i].set_title(f'{motif}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Saved: confusion_matrices.png")

def plot_multilabel_confusion_matrix(y_test, y_pred, motif_names):
    print("Creating multi-label confusion matrix aggregate...")
    
    # Get multi-label confusion matrix from sklearn
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    
    # Aggregate metrics across all labels
    tn = mcm[:, 0, 0]
    fp = mcm[:, 0, 1]
    fn = mcm[:, 1, 0]
    tp = mcm[:, 1, 1]
    
    # Create summary visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Stacked bar chart of TP, FP, TN, FN per motif
    x = np.arange(len(motif_names))
    width = 0.6
    
    axes[0].bar(x, tp, width, label='True Positive', color='#2ecc71')
    axes[0].bar(x, fp, width, bottom=tp, label='False Positive', color='#e74c3c')
    axes[0].bar(x, fn, width, bottom=tp+fp, label='False Negative', color='#f39c12')
    
    axes[0].set_xlabel('Tactical Motif', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Multi-Label Confusion Matrix (Aggregate)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(motif_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Normalized metrics per motif
    precision_per_motif = tp / (tp + fp + 1e-10)
    recall_per_motif = tp / (tp + fn + 1e-10)
    
    x_pos = np.arange(len(motif_names))
    axes[1].plot(x_pos, precision_per_motif, 'o-', label='Precision', linewidth=2, markersize=8)
    axes[1].plot(x_pos, recall_per_motif, 's-', label='Recall', linewidth=2, markersize=8)
    axes[1].set_xlabel('Tactical Motif', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Precision & Recall per Motif', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(motif_names, rotation=45, ha='right')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multilabel_confusion_aggregate.png', dpi=300, bbox_inches='tight')
    print("Saved: multilabel_confusion_aggregate.png")

def plot_performance_comparison(metrics, motif_names, y_test, y_pred):
    print("Creating performance visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Per-motif F1 scores
    f1_scores = []
    for i in range(len(motif_names)):
        f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
        f1_scores.append(f1)
    
    axes[0].barh(motif_names, f1_scores, color='skyblue')
    axes[0].set_xlabel('F1-Score', fontsize=12)
    axes[0].set_title('F1-Score per Tactical Motif', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    for i, score in enumerate(f1_scores):
        axes[0].text(score + 0.01, i, f'{score:.3f}', va='center')
    
    # Overall metrics comparison
    metric_names = ['Exact Match', 'Jaccard', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1\n(Macro)']
    metric_values = [
        metrics['exact_match'],
        metrics['jaccard'],
        metrics['precision_macro'],
        metrics['recall_macro'],
        metrics['f1_macro']
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = axes[1].bar(metric_names, metric_values, color=colors)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Overall Model Performance', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 1)
    
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    print("Saved: performance_metrics.png")

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
    
    # Build model
    model = build_cnn_model(input_shape=(8, 8, 12), num_classes=len(motif_names))
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=64)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, motif_names)
    
    # Create visualizations
    plot_training_history(history)
    plot_confusion_matrices(y_test, metrics['y_pred'], motif_names)
    plot_multilabel_confusion_matrix(y_test, metrics['y_pred'], motif_names)
    plot_performance_comparison(metrics, motif_names, y_test, metrics['y_pred'])
    
    # Save model
    model.save('chess_cnn_model.keras')
    print("\nModel saved: chess_cnn_model.keras")
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey Findings:")
    print(f"1. Exact Match Accuracy: {metrics['exact_match']:.2%}")
    print(f"2. Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"3. Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"\nGenerated Files:")
    print("  - training_history.png")
    print("  - confusion_matrices.png (10 individual per-motif matrices)")
    print("  - multilabel_confusion_aggregate.png (aggregate summary)")
    print("  - performance_metrics.png")
    print("  - chess_cnn_model.keras")

if __name__ == "__main__":
    main()
