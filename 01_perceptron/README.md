# Perceptron from Scratch

## What is it
Single layer binary classifier — the simplest neural network unit.
Foundation of all deep learning.

## Math
Forward pass:  ŷ = step(w·x + b)
Weight update: w = w + α(y - ŷ)x
Bias update:   b = b + α(y - ŷ)

## Key Insight
Works only on linearly separable data.
Fails on XOR problem — this limitation
led to invention of multi-layer networks.

## Dataset
Iris dataset — class 0 vs class 1 (binary)
Features used: 2 (sepal length, sepal width)

## Results
Accuracy: ~100% (classes are linearlInitialized empty Git repository in /Users/dhull/deep-learning-from-scratch/.git/ 53aa66b] implement perceptron from scratch with numpy
 2 files changed, 88 insertions(+)
 create mode 100644 01_perceptron/README.md
 create mode 100644 01_perceptron/perceptron.py
dhull@DHULLs-MacBook-Air deep-learning-from-scratch % y separable)