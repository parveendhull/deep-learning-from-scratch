# Backpropagation from Scratch

## What is Backpropagation
Algorithm to compute gradients of loss
with respect to every weight in the network.
Uses chain rule of calculus to propagate
error backwards from output to input.

## Chain Rule (core idea)
dL/dw = dL/da × da/dz × dz/dw

## Single Neuron
Architecture:
Input (x) → [w, b] → z → sigmoid → a → Loss

Gradients:
dL/da = -2(y - a)
da/dz = a(1 - a)        ← sigmoid derivative
dL/dw = dL/dz × x
dL/db = dL/dz × 1

Results:
Loss reduced smoothly over 100 epochs

## Multi Neuron (2 hidden, 1 output)
Architecture:
x1, x2 → [hidden neuron 1] → a1 ↘
                                    [output neuron] → a3 → Loss
x1, x2 → [hidden neuron 2] → a2 ↗

Gradients (output layer):
dL/dw5 = dL/dz3 × a1
dL/dw6 = dL/dz3 × a2

Gradients (hidden layer — chain rule back):
dL/dz1 = dL/dz3 × w5 × da1/dz1
dL/dz2 = dL/dz3 × w6 × da2/dz2

Results:
Loss reduced from 0.00959 → 0.00087
over 1000 epochs — all 9 weights updated correctly

## Key Insight
This is exactly what PyTorch autograd does
internally — automatically, for any architecture.
Understanding this manually = understanding
deep learning fundamentally.