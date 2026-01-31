
# Imports
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ( 
    accuracy_score, precision_score, recall_score, f1_score, 
    hamming_loss, jaccard_score, classification_report
    )
import chess
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Data Loading and Exploration

def load_lichess_puzzles(sample_size = 50000):
    print("Loading dataset...")
    dataset = load_dataset("lichess/chess-puzzles", split="train")

    df = pd.DataFrame(dataset)

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    print(f"Dataset loaded: {len(df)} puzzles")
    print(f"Columns: {df.columns.tolist()}")

    return df

def explore_data(df):
    print("\n Dataset Overview")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nMissing values: \n{df.isnull().sum()}")

    if 'Themes' in df.columns:
        all_themes = []
        for themes in df['Themes']:
            if isinstance(themes, list):
                all_themes.extend(themes)
            elif isinstance(themes, str):
                all_themes.extend(themes.split())
        
        themes_counts = Counter(all_themes)
        print(f"\n Tactical Motifs Distribution")
        for theme, count in themes_counts.most_common(20):
            print(f"{theme}: {count}")
        
    return df

# Feature Engineering

def fen_to_features(fen):
    
    board = chess.Board(fen)
    features = []

    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                   chess.ROOK, chess.QUEEN, chess.KING]
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in piece_types:
            count = len(board.pieces(piece_type, color))
            features.append(count)

    material_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }

    white_material = sum(material_values[pt] * len(board.pieces(pt, chess.WHITE)) for pt in piece_types)
    black_material = sum(material_values[pt] * len(board.pieces(pt, chess.BLACK)) for pt in piece_types)
    
    features.append(white_material - black_material)

    features.append(board.legal_moves.count())

    features.append(int(board.is_check()))

    features.append(int(board.has_kingside_castling_rights(chess.WHITE)))
    features.append(int(board.has_queenside_castling_rights(chess.WHITE)))
    features.append(int(board.has_kingside_castling_rights(chess.BLACK)))
    features.append(int(board.has_queenside_castling_rights(chess.BLACK)))

    attacked_count = 0
    for square in chess.SQUARES:
        if board.is_attacked_by(board.turn, square):
            attacked_count += 1
    features.append(attacked_count)
    
    return np.array(features)

def prepare_features_and_labels(df, top_n_motifs=10):
    
    print("\n Feature Engineering")

    # Select all unique themes
    all_themes = set()
    for themes in df['Themes']:
        if isinstance(themes, list):
            all_themes.update(themes)
        elif isinstance(themes, str):
            all_themes.update(themes.split())

    # Select the top N most common motifs
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
            # Extract features from FEN
            fen = row['FEN']
            features = fen_to_features(fen)
            
            # Create multi-label target
            if isinstance(row['Themes'], list):
                themes = row['Themes']
            elif isinstance(row['Themes'], str):
                themes = row['Themes'].split()
            else:
                themes = []
            label = [1 if motif in themes else 0 for motif in top_motifs]
            
            X.append(features)
            y.append(label)
            
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1} positions...")
                
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label matrix shape: {y.shape}")
    print(f"Labels per sample (avg): {y.sum(axis=1).mean():.2f}")
    
    return X, y, top_motifs

# Model Training
def train_model(X_train, y_train):

    print("Model Training")
    print("Model: Random Forest with MultiOutputClassifier")
    print("Rationale:")
    print("  - Random Forest handles complex non-linear patterns in chess positions")
    print("  - MultiOutputClassifier enables multi-label prediction")
    print("  - Provides feature importance for understanding which board features matter")
    print("  - Robust and doesn't require feature scaling")
    
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        verbose=1
    )
    
    model = MultiOutputClassifier(base_model, n_jobs=-1)
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model

# Model Evaluation

def evaluate_model(model, X_test, y_test, motif_names):

    print("\n Model Evaluation")

    # Predictions
    y_pred = model.predict(X_test)
    
    # Overall metrics
    print("\n--- Overall Metrics ---")
    hamming = hamming_loss(y_test, y_pred)
    print(f"Hamming Loss: {hamming:.4f} (lower is better)")
    
    # Exact match ratio (all labels correct)
    exact_match = accuracy_score(y_test, y_pred)
    print(f"Exact Match Ratio: {exact_match:.4f}")
    
    # Jaccard score (similarity between predicted and true label sets)
    jaccard = jaccard_score(y_test, y_pred, average='samples')
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
    
    # Per-label metrics
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

def create_visualizations(y_test, y_pred, motif_names, metrics):

    print("\n Creating Visualizations")

    # 1. Confusion Matrix for each motif
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
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
    
    # 2. Per-motif F1 scores
    plt.figure(figsize=(12, 6))
    f1_scores = []
    for i in range(len(motif_names)):
        f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
        f1_scores.append(f1)
    
    plt.barh(motif_names, f1_scores, color='skyblue')
    plt.xlabel('F1-Score')
    plt.title('F1-Score per Tactical Motif')
    plt.xlim(0, 1)
    for i, score in enumerate(f1_scores):
        plt.text(score + 0.01, i, f'{score:.3f}', va='center')
    plt.tight_layout()
    plt.savefig('f1_scores_per_motif.png', dpi=300, bbox_inches='tight')
    print("Saved: f1_scores_per_motif.png")
    
    # 3. Label distribution (true vs predicted)
    plt.figure(figsize=(12, 6))
    true_counts = y_test.sum(axis=0)
    pred_counts = y_pred.sum(axis=0)
    
    x = np.arange(len(motif_names))
    width = 0.35
    
    plt.bar(x - width/2, true_counts, width, label='True', color='lightcoral')
    plt.bar(x + width/2, pred_counts, width, label='Predicted', color='lightblue')
    plt.xlabel('Tactical Motif')
    plt.ylabel('Count')
    plt.title('True vs Predicted Label Distribution')
    plt.xticks(x, motif_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: label_distribution.png")
    
    # 4. Overall metrics comparison
    plt.figure(figsize=(10, 6))
    metric_names = ['Exact Match', 'Jaccard', 'Precision\n(Macro)', 
                   'Recall\n(Macro)', 'F1\n(Macro)']
    metric_values = [
        metrics['exact_match'],
        metrics['jaccard'],
        metrics['precision_macro'],
        metrics['recall_macro'],
        metrics['f1_macro']
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = plt.bar(metric_names, metric_values, color=colors)
    plt.ylabel('Score')
    plt.title('Overall Model Performance Metrics')
    plt.ylim(0, 1)
    
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('overall_metrics.png', dpi=300, bbox_inches='tight')
    print("Saved: overall_metrics.png")
    
    plt.show()

def main():
    print("="*80)
    print("CHESS TACTICAL MOTIF CLASSIFICATION")
    print("Assignment 3 - SEG-4180 Applied Machine Learning")
    print("="*80)
    
    # Step 1: Load and explore data
    df = load_lichess_puzzles(sample_size=10000)
    df = explore_data(df)
    
    # Step 2: Feature engineering
    X, y, motif_names = prepare_features_and_labels(df, top_n_motifs=10)
    
    # Step 3: Train-test split
    print("\n=== Train-Test Split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 4: Train model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate model
    metrics = evaluate_model(model, X_test, y_test, motif_names)
    
    # Step 6: Create visualizations
    create_visualizations(y_test, metrics['y_pred'], motif_names, metrics)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Insights:")
    print(f"1. Exact Match Accuracy: {metrics['exact_match']:.2%}")
    print(f"2. Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"3. Hamming Loss: {metrics['hamming_loss']:.4f}")

if __name__ == "__main__":
    main()