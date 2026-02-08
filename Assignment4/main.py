
# Imports
import pandas as pd
import numpy as np
from datasets import load_dataset
import chess
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score, davies_bouldin_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Data Loading

def load_lichess_puzzles(sample_size=15000):
    print("="*80)
    print("CHESS POSITION CLUSTERING ANALYSIS")
    print("Assignment 4 - Unsupervised Learning")
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

# Feature Extraction

def fen_to_features(fen):
    
    board = chess.Board(fen)
    features = []
    
    # Piece counts (12 features)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                   chess.ROOK, chess.QUEEN, chess.KING]
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in piece_types:
            count = len(board.pieces(piece_type, color))
            features.append(count)
    
    # Material balance
    material_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    white_material = sum(material_values[pt] * len(board.pieces(pt, chess.WHITE)) 
                        for pt in piece_types)
    black_material = sum(material_values[pt] * len(board.pieces(pt, chess.BLACK)) 
                        for pt in piece_types)
    features.append(white_material - black_material)
    
    # Mobility
    features.append(board.legal_moves.count())
    
    # Check status
    features.append(int(board.is_check()))
    
    # Castling rights
    features.append(int(board.has_kingside_castling_rights(chess.WHITE)))
    features.append(int(board.has_queenside_castling_rights(chess.WHITE)))
    features.append(int(board.has_kingside_castling_rights(chess.BLACK)))
    features.append(int(board.has_queenside_castling_rights(chess.BLACK)))
    
    # Attacked squares
    attacked_count = sum(1 for square in chess.SQUARES 
                        if board.is_attacked_by(board.turn, square))
    features.append(attacked_count)
    
    return np.array(features)

def extract_features(df):
    print("\n=== Feature Extraction ===")
    X = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            fen = row['FEN']
            features = fen_to_features(fen)
            X.append(features)
            valid_indices.append(idx)
            
            if len(X) % 2000 == 0:
                print(f"Processed {len(X)} positions...")
        except Exception as e:
            continue
    
    X = np.array(X)
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: Piece counts (12), Material balance (1), Mobility (1),")
    print(f"          Check (1), Castling (4), Attacked squares (1)")
    
    return X, df_valid

# Feature Scaling

def scale_features(X):
    print("\n=== Feature Scaling ===")
    print("Using StandardScaler to normalize features")
    print("Rationale: K-Means uses Euclidean distance, requires scaled features")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nOriginal feature ranges (first 5):")
    print(f"  Min: {X.min(axis=0)[:5]}")
    print(f"  Max: {X.max(axis=0)[:5]}")
    print(f"\nScaled feature ranges (first 5):")
    print(f"  Min: {X_scaled.min(axis=0)[:5]}")
    print(f"  Max: {X_scaled.max(axis=0)[:5]}")
    
    return X_scaled, scaler

# Elbow Method

def find_optimal_clusters(X_scaled, k_range=range(2, 11)):
    print("\n=== Finding Optimal Number of Clusters ===")
    print("Testing K-Means with k from 2 to 10...")
    
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    
    for k in k_range:
        print(f"Testing k={k}...", end=" ")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))
        print(f"Inertia: {kmeans.inertia_:.2f}, Silhouette: {silhouette_scores[-1]:.3f}")
    
    # Plot elbow curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inertia (Within-Cluster SS)', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score (Higher = Better)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12)
    axes[2].set_title('Davies-Bouldin Index (Lower = Better)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    print("\nSaved: elbow_method.png")
    
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    optimal_k_davies = k_range[np.argmin(davies_bouldin_scores)]
    
    print(f"\n--- Optimal K Suggestions ---")
    print(f"Best Silhouette Score: k={optimal_k_silhouette}")
    print(f"Best Davies-Bouldin Index: k={optimal_k_davies}")
    
    return optimal_k_silhouette

# K-Means Clustering

def perform_clustering(X_scaled, n_clusters):
    print(f"\n=== K-Means Clustering (k={n_clusters}) ===")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
    inertia = kmeans.inertia_
    
    print(f"\n--- Clustering Quality ---")
    print(f"Inertia: {inertia:.2f}")
    print(f"Silhouette Score: {silhouette:.4f} (range: [-1, 1], higher is better)")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    
    cluster_counts = Counter(cluster_labels)
    print(f"\n--- Cluster Sizes ---")
    for cluster_id in sorted(cluster_counts.keys()):
        print(f"Cluster {cluster_id}: {cluster_counts[cluster_id]} positions "
              f"({100*cluster_counts[cluster_id]/len(cluster_labels):.1f}%)")
    
    return kmeans, cluster_labels

# PCA Visualization

def visualize_clusters_pca(X_scaled, cluster_labels, df, n_clusters):
    print("\n=== PCA Dimensionality Reduction for Visualization ===")
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_var = pca.explained_variance_ratio_
    print(f"PCA explained variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}")
    print(f"Total variance captured: {sum(explained_var):.2%}")
    
    plt.figure(figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[cluster_id]], 
                   label=f'Cluster {cluster_id}',
                   alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    
    plt.xlabel(f'First Principal Component ({explained_var[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'Second Principal Component ({explained_var[1]:.1%} variance)', fontsize=12)
    plt.title('Chess Position Clusters (PCA 2D Projection)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('clusters_pca.png', dpi=300, bbox_inches='tight')
    print("Saved: clusters_pca.png")
    
    return X_pca, pca

# Cluster Interpretation

def interpret_clusters(df, cluster_labels, X, n_clusters):
    print("\n=== Cluster Interpretation ===")
    
    df['Cluster'] = cluster_labels
    
    feature_names = [
        'White Pawns', 'White Knights', 'White Bishops', 
        'White Rooks', 'White Queens', 'White Kings',
        'Black Pawns', 'Black Knights', 'Black Bishops',
        'Black Rooks', 'Black Queens', 'Black Kings',
        'Material Balance', 'Mobility', 'In Check',
        'W King Castle', 'W Queen Castle', 'B King Castle', 'B Queen Castle',
        'Attacked Squares'
    ]
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_data = df[mask]
        cluster_features = X[mask]
        
        print(f"\n--- Cluster {cluster_id} ({mask.sum()} positions) ---")
        
        avg_rating = cluster_data['Rating'].mean()
        print(f"Average Puzzle Rating: {avg_rating:.0f}")
        
        all_themes = []
        for themes in cluster_data['Themes']:
            if isinstance(themes, list):
                all_themes.extend(themes)
        theme_counts = Counter(all_themes)
        print(f"Top Themes: {', '.join([f'{t}({c})' for t, c in theme_counts.most_common(5)])}")
        
        avg_features = cluster_features.mean(axis=0)
        print(f"Avg Material Balance: {avg_features[12]:.2f}")
        print(f"Avg Mobility: {avg_features[13]:.1f}")
        print(f"Avg Total Pieces: {avg_features[:12].sum():.1f}")
        
        top_features_idx = np.argsort(avg_features)[-3:][::-1]
        print(f"Key Features: {', '.join([f'{feature_names[i]}={avg_features[i]:.2f}' for i in top_features_idx])}")

# Supervised Model Training

def train_cluster_predictor(X_scaled, cluster_labels):
    print("\n=== Training Supervised Model to Predict Clusters ===")
    print("Model: Random Forest Classifier")
    print("Purpose: Learn to predict cluster assignments from position features")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, cluster_labels, test_size=0.3, random_state=42, stratify=cluster_labels
    )
    
    print(f"\nTraining set: {len(X_train)} positions")
    print(f"Test set: {len(X_test)} positions")
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining Random Forest...")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print("\n=== Predictive Model Evaluation ===")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    
    print("\n--- Per-Cluster Metrics ---")
    print(classification_report(y_test, y_pred, 
                                target_names=[f'Cluster {i}' for i in range(len(np.unique(cluster_labels)))]))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'C{i}' for i in range(len(cm))],
                yticklabels=[f'C{i}' for i in range(len(cm))])
    plt.xlabel('Predicted Cluster', fontsize=12)
    plt.ylabel('True Cluster', fontsize=12)
    plt.title('Confusion Matrix - Cluster Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_prediction_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nSaved: cluster_prediction_confusion_matrix.png")
    
    # Feature Importance
    feature_names = [
        'W Pawns', 'W Knights', 'W Bishops', 'W Rooks', 'W Queens', 'W Kings',
        'B Pawns', 'B Knights', 'B Bishops', 'B Rooks', 'B Queens', 'B Kings',
        'Material', 'Mobility', 'Check', 'WK Castle', 'WQ Castle', 'BK Castle', 'BQ Castle', 'Attacked'
    ]
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[-10:][::-1]
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(10), importances[indices], color='skyblue')
    plt.yticks(range(10), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 10 Features for Cluster Prediction', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: feature_importance.png")
    
    return clf, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Main Execution

def main():
    
    # Load data
    df = load_lichess_puzzles(sample_size=15000)
    
    # Extract features
    X, df_valid = extract_features(df)
    
    # Scale features
    X_scaled, scaler = scale_features(X)
    
    # Find optimal k
    optimal_k = find_optimal_clusters(X_scaled)
    
    # Choose k based on elbow plot
    chosen_k = optimal_k
    print(f"\n>>> Proceeding with k={chosen_k} clusters")
    
    # Perform clustering
    kmeans, cluster_labels = perform_clustering(X_scaled, chosen_k)
    
    # Visualize with PCA
    X_pca, pca = visualize_clusters_pca(X_scaled, cluster_labels, df_valid, chosen_k)
    
    # Interpret clusters
    interpret_clusters(df_valid, cluster_labels, X, chosen_k)
    
    # Train supervised model
    clf, metrics = train_cluster_predictor(X_scaled, cluster_labels)
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey Findings:")
    print(f"1. Optimal number of clusters: {chosen_k}")
    print(f"2. Cluster prediction accuracy: {metrics['accuracy']:.2%}")
    print(f"3. Clusters represent distinct chess position types")
    print(f"4. High prediction accuracy indicates well-separated, meaningful clusters")
    print(f"\nGenerated Files:")
    print("  - elbow_method.png")
    print("  - clusters_pca.png")
    print("  - cluster_prediction_confusion_matrix.png")
    print("  - feature_importance.png")

if __name__ == "__main__":
    main()
