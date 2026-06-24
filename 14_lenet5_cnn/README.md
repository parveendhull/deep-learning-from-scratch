# LeNet-5 — Classic CNN Architecture

## What is it
One of the earliest Convolutional Neural Networks,
introduced by Yann LeCun (1998) for handwritten
digit recognition. Was used commercially to read
bank cheques.

## Architecture
Input(32×32×1)
→ Conv2D(6 filters, 5×5, tanh)
→ AvgPool(2×2)
→ Conv2D(16 filters, 5×5, tanh)
→ AvgPool(2×2)
→ Flatten
→ Dense(120, tanh)
→ Dense(84, tanh)
→ Dense(10, softmax)

## Key Historical Choices
- Tanh activation (ReLU didn't exist yet in 1998)
- Average Pooling (not Max Pooling)
- Input padded to 32×32 (original design choice)

## Why CNN over Plain Neural Network
- Same filter slides across entire image (parameter sharing)
- Detects same pattern regardless of position
  (translation invariance)
- Far fewer parameters than a fully connected
  network of similar capacity

## Hierarchical Feature Learning
Layer 1 → simple edges (horizontal, vertical, diagonal)
Layer 2 → combinations of edges → curves, corners
Deeper layers → complex patterns, shapes, digits

## Results
| Setup | Test Accuracy |
|-------|----------------|
| Buggy (double normalization /255 twice) | 97.41% |
| Fixed (single normalization)            | 98.71% |

## Key Debugging Lesson
Pixel values were divided by 255 twice — once during
cast, once during padding step — making values
extremely small (~0.0039 max instead of 1.0).
This made gradients too small for the model to
learn effectively. Fixing normalization improved
accuracy by 1.3 percentage points.