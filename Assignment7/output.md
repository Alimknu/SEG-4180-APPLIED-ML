================================================================================
LOADING CIFAR-10 DATASET
================================================================================
Total training samples: 50000
Labeled samples: 1000 (max 1000)
Unlabeled samples: 2000 (max 2000)
Test samples: 10000 (max 1000)

Labeled shape: (1000, 224, 224, 3)
Unlabeled shape: (2000, 224, 224, 3)
Test shape: (1000, 224, 224, 3)

================================================================================
INITIAL SUPERVISED TRAINING (10% labeled data)
================================================================================
Epoch 1/10
16/16 ━━━━━━━━━━━━━━━━━━━━ 29s 2s/step - accuracy: 0.4280 - loss: 1.7830 - val_accuracy: 0.7390 - val_loss: 0.7446
Epoch 2/10
16/16 ━━━━━━━━━━━━━━━━━━━━ 25s 2s/step - accuracy: 0.7100 - loss: 0.8531 - val_accuracy: 0.8190 - val_loss: 0.5389
Epoch 3/10
16/16 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.7580 - loss: 0.7195 - val_accuracy: 0.8160 - val_loss: 0.5051
Epoch 4/10
16/16 ━━━━━━━━━━━━━━━━━━━━ 25s 2s/step - accuracy: 0.7970 - loss: 0.5981 - val_accuracy: 0.8380 - val_loss: 0.4694
Epoch 5/10
16/16 ━━━━━━━━━━━━━━━━━━━━ 25s 2s/step - accuracy: 0.8240 - loss: 0.5132 - val_accuracy: 0.8510 - val_loss: 0.4310
Epoch 6/10
16/16 ━━━━━━━━━━━━━━━━━━━━ 25s 2s/step - accuracy: 0.8310 - loss: 0.4914 - val_accuracy: 0.8320 - val_loss: 0.4717
Epoch 7/10
16/16 ━━━━━━━━━━━━━━━━━━━━ 25s 2s/step - accuracy: 0.8590 - loss: 0.4151 - val_accuracy: 0.8480 - val_loss: 0.4173
Epoch 8/10
16/16 ━━━━━━━━━━━━━━━━━━━━ 25s 2s/step - accuracy: 0.8590 - loss: 0.3903 - val_accuracy: 0.8550 - val_loss: 0.4266
Epoch 9/10
16/16 ━━━━━━━━━━━━━━━━━━━━ 25s 2s/step - accuracy: 0.8850 - loss: 0.3480 - val_accuracy: 0.8460 - val_loss: 0.4326
Epoch 10/10
16/16 ━━━━━━━━━━━━━━━━━━━━ 25s 2s/step - accuracy: 0.8710 - loss: 0.3411 - val_accuracy: 0.8510 - val_loss: 0.4216

================================================================================
PSEUDO-LABELING ITERATION 1
================================================================================

Generating pseudo-labels with threshold 0.95...
32/32 ━━━━━━━━━━━━━━━━━━━━ 26s 793ms/step
Retained 1183 / 2000 pseudo-labeled samples (59.2%)
Epoch 1/5
35/35 ━━━━━━━━━━━━━━━━━━━━ 50s 1s/step - accuracy: 0.9052 - loss: 0.2819 - val_accuracy: 0.8580 - val_loss: 0.6308
Epoch 2/5
35/35 ━━━━━━━━━━━━━━━━━━━━ 45s 1s/step - accuracy: 0.9505 - loss: 0.1574 - val_accuracy: 0.8520 - val_loss: 0.6554
Epoch 3/5
35/35 ━━━━━━━━━━━━━━━━━━━━ 46s 1s/step - accuracy: 0.9542 - loss: 0.1335 - val_accuracy: 0.8680 - val_loss: 0.5597
Epoch 4/5
35/35 ━━━━━━━━━━━━━━━━━━━━ 45s 1s/step - accuracy: 0.9684 - loss: 0.1024 - val_accuracy: 0.8740 - val_loss: 0.4451
Epoch 5/5
35/35 ━━━━━━━━━━━━━━━━━━━━ 45s 1s/step - accuracy: 0.9821 - loss: 0.0693 - val_accuracy: 0.8700 - val_loss: 0.4605

================================================================================
PSEUDO-LABELING ITERATION 2
================================================================================

Generating pseudo-labels with threshold 0.95...
32/32 ━━━━━━━━━━━━━━━━━━━━ 26s 780ms/step
Retained 1590 / 2000 pseudo-labeled samples (79.5%)
Epoch 1/5
41/41 ━━━━━━━━━━━━━━━━━━━━ 58s 1s/step - accuracy: 0.9452 - loss: 0.1721 - val_accuracy: 0.8530 - val_loss: 0.6269
Epoch 2/5
41/41 ━━━━━━━━━━━━━━━━━━━━ 53s 1s/step - accuracy: 0.9660 - loss: 0.1077 - val_accuracy: 0.8690 - val_loss: 0.5374
Epoch 3/5
41/41 ━━━━━━━━━━━━━━━━━━━━ 53s 1s/step - accuracy: 0.9741 - loss: 0.0871 - val_accuracy: 0.8760 - val_loss: 0.4996
Epoch 4/5
41/41 ━━━━━━━━━━━━━━━━━━━━ 52s 1s/step - accuracy: 0.9772 - loss: 0.0696 - val_accuracy: 0.8700 - val_loss: 0.5358
Epoch 5/5
41/41 ━━━━━━━━━━━━━━━━━━━━ 53s 1s/step - accuracy: 0.9795 - loss: 0.0753 - val_accuracy: 0.8770 - val_loss: 0.5277

================================================================================
FINAL EVALUATION
================================================================================

Classification Report:
              precision    recall  f1-score   support

           0     0.8476    0.8641    0.8558       103
           1     0.9540    0.9326    0.9432        89
           2     0.8586    0.8500    0.8543       100
           3     0.7200    0.8738    0.7895       103
           4     0.8953    0.8556    0.8750        90
           5     0.8493    0.7209    0.7799        86
           6     0.8632    0.9018    0.8821       112
           7     0.9583    0.9020    0.9293       102
           8     0.9604    0.9151    0.9372       106
           9     0.9099    0.9266    0.9182       109

    accuracy                         0.8770      1000
   macro avg     0.8817    0.8742    0.8764      1000
weighted avg     0.8813    0.8770    0.8777      1000

Saved: confusion_matrix.png
Saved: training_curves.png

Model saved: Assignment7/resnet50_semi_final.keras