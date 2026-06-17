# Activation Functions and Their Derivatives

## What is Covered
Implementation and visualization of 4 activation
functions and their derivatives, used to understand
how each affects gradient flow during backpropagation.

## Functions Implemented

### Sigmoid
f(x) = 1 / (1 + e^-x)
f'(x) = f(x)(1 - f(x))
Output range: (0, 1)
Max derivative: 0.25 (at x=0)

### Tanh
f(x) = tanh(x)
f'(x) = 1 - tanh²(x)
Output range: (-1, 1)
Max derivative: 1.0 (at x=0)

### ReLU
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
Output range: [0, ∞)
Derivative: constant 1 for x > 0

### Leaky ReLU
f(x) = x if x > 0, else αx
f'(x) = 1 if x > 0, else α
Output range: (-∞, ∞)
Fixes ReLU's "dying neuron" problem for negative inputs

## Key Observation
Sigmoid has the smallest derivative range (max 0.25),
making it most prone to vanishing gradients.
Far from x=0, both Sigmoid and Tanh derivatives
approach zero — gradients "die" during backprop
through deep networks.

ReLU avoids this in the positive region —
derivative stays exactly 1, regardless of x.
This is why ReLU enabled training of much deeper
networks than Sigmoid/Tanh allowed.

## Connection to Vanishing Gradient Problem
This directly explains why deep networks with
Sigmoid/Tanh fail to train — multiplying many
small derivatives (≤0.25) across layers during
backprop shrinks the gradient toward zero.

See: 07_vanishing_gradient/ for full demonstration.

## Visualization
4 subplots comparing function (blue) and
derivative (red, dashed) for each activation.