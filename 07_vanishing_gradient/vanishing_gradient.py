import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Show how gradients shrink layer by layer
layers = 10
z = 1.0  # same input for all layers

gradients = []
grad = 1.0

for i in range(layers):
    grad = grad * sigmoid_derivative(z)
    gradients.append(grad)
    print(f'Layer {i+1}: gradient = {grad:.8f}')

# Plot
plt.figure(figsize=(8, 4))
plt.plot(range(1, layers+1), gradients, marker='o', color='red')
plt.title('Vanishing Gradient — Sigmoid')
plt.xlabel('Layer')
plt.ylabel('Gradient magnitude')
plt.tight_layout()
plt.show()


def relu_derivative(z):
    return 1 if z > 0 else 0

grad_relu = 1.0
gradients_relu = []

for i in range(layers):
    grad_relu = grad_relu * relu_derivative(z)
    gradients_relu.append(grad_relu)

# Compare plot
plt.figure(figsize=(8, 4))
plt.plot(range(1, layers+1), gradients,      marker='o', label='Sigmoid', color='red')
plt.plot(range(1, layers+1), gradients_relu, marker='o', label='ReLU',    color='green')
plt.title('Vanishing Gradient — Sigmoid vs ReLU')
plt.xlabel('Layer')
plt.ylabel('Gradient magnitude')
plt.legend()
plt.tight_layout()
plt.show()
