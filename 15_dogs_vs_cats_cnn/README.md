# Dogs vs Cats — CNN Image Classification

## Problem
Binary image classification — distinguish dogs
from cats using a CNN trained from scratch
(no transfer learning).

## Architecture
data_augmentation (RandomFlip, RandomRotation, RandomZoom)
→ Conv2D(32) → MaxPool
→ Conv2D(64) → MaxPool
→ Conv2D(128) → MaxPool
→ GlobalMaxPool2D
→ Dense(128) → Dropout(0.3)
→ Dense(64)  → Dropout(0.2)
→ Dense(1, sigmoid)

## Why GlobalMaxPool2D over Flatten
Flatten preserves spatial position info and creates
a large number of parameters going into the dense
layers. GlobalMaxPool2D extracts the strongest
feature signal from each filter map regardless of
position — fewer parameters, less overfitting risk.

## Why tf.keras.layers for Augmentation (not ImageDataGenerator)
ImageDataGenerator is the older, deprecated approach.
tf.keras.layers.RandomFlip/RandomRotation/RandomZoom:
- Run as part of the model graph — GPU accelerated
- Apply only during training (auto-disabled at inference)
- Are the current recommended approach in modern Keras

## Experimental Results (systematic improvements)
| Model                      | Val Accuracy | Notes |
|-----------------------------|-------------:|-------|
| Baseline CNN                | 84.70%       | |
| + Dropout                   | 86.84%       | |
| + GlobalMaxPooling2D         | 87.52%       | |
| + EarlyStopping              | 88.86%       | |
| + Data Augmentation          | **89.88%**   | Best — train/val gap small |
| + BatchNormalization (tried) | 76.00%       | Overfit — train acc 95.9%, rejected |

## Key Insight — BatchNorm Made Things Worse
Adding BatchNormalization caused severe overfitting
(train 95.9% vs val 76%) despite being a common
regularization technique. Likely causes:
- Small dataset → unstable batch statistics
- Interaction with existing Dropout layers
- Train vs inference mode discrepancy amplified
  on limited data

This was kept out of the final model. Lesson:
adding a "standard" technique doesn't guarantee
improvement — always validate empirically rather
than assuming theoretical benefits transfer directly.

## Final Result
Best Epoch: 24
Train Accuracy: 88.63%
Validation Accuracy: 89.88%
Train Loss: 0.2684
Validation Loss: 0.2367

Train and validation metrics are close —
indicates good generalization, minimal overfitting.