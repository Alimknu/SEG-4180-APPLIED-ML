

# ================================================================================
# Assignment 10 - AutoML Pipeline for Chess Tactics Classification
# Tool: LazyPredict (sklearn-based AutoML - no Dask, Python 3.13 compatible)
# Project: Chessplained - Chess Optimal Move Detector with Explanations
# ================================================================================

# -------------------- Imports and Environment Setup --------------------
import warnings
warnings.filterwarnings('ignore')

import random
import chess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lazypredict.Supervised import LazyClassifier

# -------------------- Configuration --------------------
DATASET_SIZE = 500
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -------------------- Synthetic Chess Data Generation --------------------
def random_board():
    """Generate a random chess board by making random legal moves."""
    board = chess.Board()
    for _ in range(random.randint(10, 30)):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    return board

# -------------------- Feature Extraction --------------------
def extract_features(board, move):
    """Extract features from the board and move, from the mover's perspective."""
    features = {
        'is_check': int(board.gives_check(move)),
        'is_capture': int(board.is_capture(move)),
        'material_balance': sum([piece.piece_type for piece in board.piece_map().values()]),
        'num_attackers': len(board.attackers(board.turn, board.king(not board.turn))) if board.gives_check(move) else 0,
    }
    return features

# -------------------- Tactic Label Assignment --------------------
def assign_tactic(features):
    """Assign a synthetic tactic label based on extracted features."""
    if features['is_check'] and features['num_attackers'] > 1:
        return 'fork'
    elif features['is_check']:
        return 'check'
    elif features['is_capture']:
        return 'capture'
    else:
        return 'none'

# -------------------- Dataset Creation --------------------
print("=" * 80)
print("CHESS TACTICS CLASSIFIER - AUTOML PIPELINE (LazyPredict)")
print("Assignment 10 - Chessplained Project")
print("=" * 80)

rows = []
for _ in range(DATASET_SIZE):
    board = random_board()
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        continue  # Skip boards in game-over states
    move = random.choice(legal_moves)
    features = extract_features(board, move)
    label = assign_tactic(features)
    row = features.copy()
    row['label'] = label
    rows.append(row)

# -------------------- Preprocessing --------------------
df = pd.DataFrame(rows)
df.to_csv('Assignment10/chess_tactics_sample.csv', index=False)
print(f'\nDataset saved: {len(df)} samples')
print(f'Features: {[c for c in df.columns if c != "label"]}')
print(f'\nLabel distribution:\n{df["label"].value_counts().to_string()}')

# -------------------- AutoML Pipeline (LazyPredict) --------------------

# Explicit label encoding for robustness
from sklearn.preprocessing import LabelEncoder
X = df.drop('label', axis=1)
y = df['label']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f'\nTraining set: {len(X_train)} samples | Test set: {len(X_test)} samples')
print('\nRunning LazyPredict AutoML - evaluating all classifiers...\n')

clf = LazyClassifier(verbose=1, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# -------------------- Results --------------------
print("\n" + "=" * 80)
print("AUTOML RESULTS - ALL MODELS RANKED BY ACCURACY")
print("=" * 80)
print(models.to_string())

best_model_name = models.index[0]
print(f'\nBest Model: {best_model_name}')
print(f'Best Accuracy: {models["Accuracy"].iloc[0]:.3f}')
print(f'Best F1 Score: {models["F1 Score"].iloc[0]:.3f}')

# -------------------- Classification Report (Best Model) --------------------
# Re-train the best model directly to generate a full classification report
from sklearn.utils import all_estimators

estimators = dict(all_estimators(type_filter='classifier'))
if best_model_name in estimators:
    best_clf = estimators[best_model_name]()
    best_clf.fit(X_train, y_train)
    y_pred_best = best_clf.predict(X_test)
    print(f'\nClassification Report ({best_model_name}):')
    print(classification_report(y_test, y_pred_best, digits=3, zero_division=0))