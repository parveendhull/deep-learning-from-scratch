import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.1):
    return np.where(x > 0, 1.0, alpha)

x = np.linspace(-10, 10, 400)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.ravel()

functions = [
    ('Sigmoid', sigmoid, sigmoid_derivative),
    ('Tanh', tanh, tanh_derivative),
    ('ReLU', relu, relu_derivative),
    ('Leaky ReLU', leaky_relu, leaky_relu_derivative)
]

for i, (name, func, deriv) in enumerate(functions):
    axs[i].plot(x, func(x), label=f'{name} f(x)', color='blue', lw=2.5)

    axs[i].plot(x, deriv(x), label=f"Derivative f'(x)", color='red', linestyle='--', lw=2)

    axs[i].set_title(f'{name} & Its Derivative', fontsize=14, fontweight='bold')
    axs[i].set_xlabel('x', fontsize=12)
    axs[i].set_ylabel('y', fontsize=12)
    axs[i].grid(True, linestyle=':', alpha=0.6)
    axs[i].legend(fontsize=11)


    axs[i].axhline(0, color='black', linewidth=0.5, alpha=0.5)
    axs[i].axvline(0, color='black', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.show()
