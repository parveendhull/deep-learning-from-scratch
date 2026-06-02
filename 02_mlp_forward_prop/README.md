# MLP Forward Propagation from Scratch

## Architecture
Input layer  → 3 neurons
Hidden layer → 4 neurons (ReLU)
Output layer → 1 neuron  (Sigmoid)

## Math
z1 = X · w1 + b1
a1 = ReLU(z1)
z2 = a1 · w2 + b2
a2 = sigmoid(z2)

## Cache
z1, a1, z2, a2 stored for backprop use