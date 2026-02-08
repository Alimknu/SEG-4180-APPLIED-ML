================================================================================
CHESS POSITION CLUSTERING ANALYSIS
Assignment 4 - Unsupervised Learning
================================================================================

Loading Lichess dataset...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Dataset loaded: 15000 puzzles
Columns: ['PuzzleId', 'GameId', 'FEN', 'Moves', 'Rating', 'RatingDeviation', 'Popularity', 'NbPlays', 'Themes', 'OpeningTags']

--- Sample Data ---
         Rating                                             Themes                                                FEN
96671      1796  [attraction, endgame, exposedKing, fork, long,...  7r/1b3k2/p3p3/1ppq1pQ1/3P4/1P3PP1/P7/2RR2K1 w ...
2965920     555                    [endgame, mate, mateIn2, short]  5rk1/p5pp/2p2r2/3p3B/4b2Q/P7/2P3PP/2q1R2K w - ...
3199220    1095                  [advantage, endgame, fork, short]  6k1/4r2p/3NP1p1/2pP2p1/1pPb2P1/pP4KP/P7/5R2 w ...
1688664     884                    [endgame, mate, mateIn2, short]      2R4R/p5p1/6k1/7p/5B1K/5PP1/PP4qP/8 w - - 4 33
5015976    1177                  [advantage, endgame, fork, short]  6k1/p3qpp1/1p3n1p/2pp4/4rP2/3Q2NP/PPP3P1/3R2K1...

=== Feature Extraction ===
Processed 2000 positions...
Processed 4000 positions...
Processed 6000 positions...
Processed 8000 positions...
Processed 10000 positions...
Processed 12000 positions...
Processed 14000 positions...

Feature matrix shape: (15000, 20)
Features: Piece counts (12), Material balance (1), Mobility (1),
          Check (1), Castling (4), Attacked squares (1)

=== Feature Scaling ===
Using StandardScaler to normalize features
Rationale: K-Means uses Euclidean distance, requires scaled features

Original feature ranges (first 5):
  Min: [0 0 0 0 0]
  Max: [8 2 2 2 2]

Scaled feature ranges (first 5):
  Min: [-3.03579774 -0.9076711  -1.01028255 -1.87517927 -1.50388708]
  Max: [1.92134862 1.97412941 1.73978337 0.86284166 2.8180415 ]

=== Finding Optimal Number of Clusters ===
Testing K-Means with k from 2 to 10...
Testing k=2... Inertia: 210266.73, Silhouette: 0.214
Testing k=3... Inertia: 171639.84, Silhouette: 0.252
Testing k=4... Inertia: 152837.86, Silhouette: 0.266
Testing k=5... Inertia: 140843.68, Silhouette: 0.189
Testing k=6... Inertia: 128160.95, Silhouette: 0.202
Testing k=7... Inertia: 119431.29, Silhouette: 0.193
Testing k=8... Inertia: 113903.74, Silhouette: 0.178
Testing k=9... Inertia: 108193.87, Silhouette: 0.187
Testing k=10... Inertia: 104895.74, Silhouette: 0.190

Saved: elbow_method.png

--- Optimal K Suggestions ---
Best Silhouette Score: k=4
Best Davies-Bouldin Index: k=4

>>> Proceeding with k=4 clusters

=== K-Means Clustering (k=4) ===

--- Clustering Quality ---
Inertia: 152837.86
Silhouette Score: 0.2657 (range: [-1, 1], higher is better)
Davies-Bouldin Index: 1.4511 (lower is better)

--- Cluster Sizes ---
Cluster 0: 7924 positions (52.8%)
Cluster 1: 1296 positions (8.6%)
Cluster 2: 1857 positions (12.4%)
Cluster 3: 3923 positions (26.2%)

=== PCA Dimensionality Reduction for Visualization ===
PCA explained variance: PC1=33.49%, PC2=13.56%
Total variance captured: 47.06%
Saved: clusters_pca.png

=== Cluster Interpretation ===

--- Cluster 0 (7924 positions) ---
Average Puzzle Rating: 1518
Top Themes: middlegame(5166), short(4226), advantage(2916), crushing(2637), endgame(2561)
Avg Material Balance: -0.09
Avg Mobility: 37.4
Avg Total Pieces: 21.0
Key Features: Mobility=37.43, Attacked Squares=36.26, Black Pawns=5.41

--- Cluster 1 (1296 positions) ---
Average Puzzle Rating: 1395
Top Themes: middlegame(762), short(693), advantage(558), opening(530), mate(417)
Avg Material Balance: -0.15
Avg Mobility: 33.6
Avg Total Pieces: 26.9
Key Features: Attacked Squares=34.65, Mobility=33.60, Black Pawns=6.60

--- Cluster 2 (1857 positions) ---
Average Puzzle Rating: 1324
Top Themes: mate(1156), endgame(1062), short(801), middlegame(777), mateIn1(612)
Avg Material Balance: -0.20
Avg Mobility: 3.0
Avg Total Pieces: 17.9
Key Features: Attacked Squares=34.56, Black Pawns=4.60, White Pawns=4.56

--- Cluster 3 (3923 positions) ---
Average Puzzle Rating: 1487
Top Themes: endgame(3879), crushing(2527), short(2067), long(1076), mate(753)
Avg Material Balance: -0.01
Avg Mobility: 18.3
Avg Total Pieces: 11.8
Key Features: Attacked Squares=22.32, Mobility=18.29, Black Pawns=3.54

=== Training Supervised Model to Predict Clusters ===
Model: Random Forest Classifier
Purpose: Learn to predict cluster assignments from position features

Training set: 10500 positions
Test set: 4500 positions

Training Random Forest...

=== Predictive Model Evaluation ===
Accuracy: 0.9840
Precision (weighted): 0.9840
Recall (weighted): 0.9840
F1-Score (weighted): 0.9840

--- Per-Cluster Metrics ---
              precision    recall  f1-score   support

   Cluster 0       0.98      0.99      0.99      2377
   Cluster 1       1.00      0.98      0.99       389
   Cluster 2       0.99      0.99      0.99       557
   Cluster 3       0.98      0.97      0.97      1177

    accuracy                           0.98      4500
   macro avg       0.99      0.98      0.99      4500
weighted avg       0.98      0.98      0.98      4500


Saved: cluster_prediction_confusion_matrix.png
Saved: feature_importance.png

================================================================================
ANALYSIS COMPLETE
================================================================================

Key Findings:
1. Optimal number of clusters: 4
2. Cluster prediction accuracy: 98.40%
3. Clusters represent distinct chess position types
4. High prediction accuracy indicates well-separated, meaningful clusters

Generated Files:
  - elbow_method.png
  - clusters_pca.png
  - cluster_prediction_confusion_matrix.png
  - feature_importance.png
