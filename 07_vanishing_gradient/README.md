# Vanishing Gradient Problem

## What is it
During backpropagation, gradients get
multiplied layer by layer going backwards.
If gradients are small (< 1), they shrink
exponentially — deeper layers learn nothing.

## Why Sigmoid Causes it
Sigmoid derivative: s(1 - s)
Maximum value = 0.25 (at z=0)

So every layer multiplies gradient by ≤ 0.25
10 layers → 0.25^10 = 0.000000095

Gradient practically becomes zero — weights stop updating.

## Demonstration
Layer 1:  gradient = 0.19661193
Layer 2:  gradient = 0.03865824
Layer 3:  gradient = 0.00760612
Layer 4:  gradient = 0.00149618
Layer 5:  gradient = 0.00029432
Layer 6:  gradient = 0.00005791
Layer 7:  gradient = 0.00001139
Layer 8:  gradient = 0.00000224
Layer 9:  gradient = 0.00000044
Layer 10: gradient = 0.00000009

## Why ReLU Solves it
ReLU derivative = 1 (for z > 0)
Gradient stays 1.0 across all layers
No shrinking — deep networks can learn

## Solutions
1. Use ReLU instead of Sigmoid
2. Batch Normalization
3. Residual connections (ResNet)
4. Better weight initialization (He, Xavier)

## Key Insight
This is why deep networks with sigmoid
failed before ReLU was widely adopted.
ReLU made modern deep learning possible.