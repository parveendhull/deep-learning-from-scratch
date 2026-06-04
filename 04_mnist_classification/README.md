# MNIST Handwritten Digit Classification

## Architecture
Input  → Flatten (28x28 → 784)
Layer1 → Dense 128 (ReLU) + Dropout(0.3)
Layer2 → Dense 32  (ReLU) + Dropout(0.2)
Output → Dense 10  (Softmax)

## Math
Softmax: σ(z)i = e^zi / Σe^zj
Loss: sparse_categorical_crossentropy

## Key Concepts
- Flatten converts 2D image to 1D vector
- Softmax converts scores to probabilities
- argmax converts probabilities to class label
- Dropout prevents overfitting

## Results
Accuracy: 97.65% on test set
Epochs: 20, Batch size: 32
Device: Apple M4 (CPU) — 1s per epoch