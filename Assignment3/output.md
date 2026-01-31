Loading dataset...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Dataset loaded: 10000 puzzles
Columns: ['PuzzleId', 'GameId', 'FEN', 'Moves', 'Rating', 'RatingDeviation', 'Popularity', 'NbPlays', 'Themes', 'OpeningTags']

 Dataset Overview
        PuzzleId             GameId  ...                                             Themes                                        OpeningTags
3251829    ZVI6F        DZXx8CZr#25  ...    [advantage, deflection, fork, long, middlegame]  [French_Defense, French_Defense_Winawer_Variat...
1370210    Ex7jY        NRugT2LN#95  ...           [crushing, defensiveMove, endgame, long]                                               None
2496511    RFIwm  cm2NjPWh/black#56  ...  [clearance, crushing, middlegame, pin, sacrifi...                                               None
2302947    P8DrW       DSdIEpYL#127  ...           [crushing, endgame, queenEndgame, short]                                               None
4322633    lBAI3        8qGuUKuv#93  ...  [crushing, discoveredAttack, discoveredCheck, ...                                               None

[5 rows x 10 columns]

Dataset shape: (10000, 10)

Missing values:
PuzzleId              0
GameId                0
FEN                   0
Moves                 0
Rating                0
RatingDeviation       0
Popularity            0
NbPlays               0
Themes                0
OpeningTags        7947
dtype: int64

 Tactical Motifs Distribution
short: 5101
endgame: 4929
middlegame: 4572
crushing: 3911
mate: 3070
advantage: 2956
long: 2658
master: 1422
oneMove: 1382
mateIn1: 1378
fork: 1312
mateIn2: 1306
kingsideAttack: 899
veryLong: 816
sacrifice: 741
pin: 622
defensiveMove: 601
advancedPawn: 578
discoveredAttack: 514
rookEndgame: 510

 Feature Engineering
Top 10 motifs: ['short', 'endgame', 'middlegame', 'crushing', 'mate', 'advantage', 'long', 'master', 'oneMove', 'mateIn1']
Processed 2989000 positions...
Processed 3402000 positions...
Processed 84000 positions...
Processed 366000 positions...
Processed 347000 positions...
Processed 1240000 positions...
Processed 2321000 positions...

Feature matrix shape: (10000, 20)
Label matrix shape: (10000, 10)
Labels per sample (avg): 3.14

=== Train-Test Split ===
Training set: 8000 samples
Test set: 2000 samples
Model Training
Model: Random Forest with MultiOutputClassifier
Rationale:
  - Random Forest handles complex non-linear patterns in chess positions
  - MultiOutputClassifier enables multi-label prediction
  - Provides feature importance for understanding which board features matter
  - Robust and doesn't require feature scaling

Training model...
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.1s finished
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.1s finished
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.1s finished
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.1s finished
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.1s finished
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.1s finished
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.1s finished
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.1s finished
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.0s finished
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.0s finished
Training complete!

 Model Evaluation
[Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=20)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=20)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=20)]: Done 100 out of 100 | elapsed:    0.0s finished
[Parallel(n_jobs=20)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=20)]: Done 100 out of 100 | elapsed:    0.0s finished
[Parallel(n_jobs=20)]: Done 100 out of 100 | elapsed:    0.0s finished
[Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=20)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=20)]: Done 100 out of 100 | elapsed:    0.0s finished
[Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=20)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=20)]: Done 100 out of 100 | elapsed:    0.0s finished
[Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=20)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=20)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=20)]: Done 100 out of 100 | elapsed:    0.0s finished
[Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=20)]: Done 100 out of 100 | elapsed:    0.0s finished
[Parallel(n_jobs=20)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=20)]: Done 100 out of 100 | elapsed:    0.0s finished
[Parallel(n_jobs=20)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=20)]: Done 100 out of 100 | elapsed:    0.0s finished
[Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
[Parallel(n_jobs=20)]: Done  10 tasks      | elapsed:    0.0s
[Parallel(n_jobs=20)]: Done 100 out of 100 | elapsed:    0.0s finished

--- Overall Metrics ---
Hamming Loss: 0.2376 (lower is better)
Exact Match Ratio: 0.1440
Jaccard Score: 0.4994

--- Averaged Metrics ---
Macro-averaged Precision: 0.5320
Macro-averaged Recall: 0.5341
Macro-averaged F1-Score: 0.5303

Micro-averaged Precision: 0.6201
Micro-averaged Recall: 0.6268
Micro-averaged F1-Score: 0.6235

--- Per-Motif Metrics ---
short                - Precision: 0.542, Recall: 0.632, F1: 0.584, Support: 1012
endgame              - Precision: 0.972, Recall: 0.983, F1: 0.977, Support: 989
middlegame           - Precision: 0.915, Recall: 0.952, F1: 0.933, Support: 899
crushing             - Precision: 0.638, Recall: 0.508, F1: 0.565, Support: 778
mate                 - Precision: 0.556, Recall: 0.484, F1: 0.518, Support: 626
advantage            - Precision: 0.461, Recall: 0.617, F1: 0.528, Support: 585
long                 - Precision: 0.328, Recall: 0.297, F1: 0.312, Support: 546
master               - Precision: 0.166, Recall: 0.139, F1: 0.151, Support: 273
oneMove              - Precision: 0.367, Recall: 0.363, F1: 0.365, Support: 284
mateIn1              - Precision: 0.375, Recall: 0.366, F1: 0.371, Support: 284

 Creating Visualizations
Saved: confusion_matrices.png
Saved: f1_scores_per_motif.png
Saved: label_distribution.png
Saved: overall_metrics.png

================================================================================
ANALYSIS COMPLETE
================================================================================

Key Insights:
1. Exact Match Accuracy: 14.40%
2. Macro F1-Score: 0.5303
3. Hamming Loss: 0.2376

Next steps for Chessplained:
- Integrate this model to tag moves with tactical motifs
- Use motif predictions to generate human-readable explanations
- Combine with chess engine evaluations for comprehensive move analysis