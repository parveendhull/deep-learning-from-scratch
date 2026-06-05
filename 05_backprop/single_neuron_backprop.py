import numpy as np
import matplotlib.pyplot as plt

# Input and target
x = 2
y = 1

# Initialize weights, bias, learning rate
w = 0.5
b = 1.0
lr = 0.1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

losses = []

for i in range(101):
    # Forward pass
    z = x * w + b
    a = sigmoid(z)
    L = (y - a) ** 2
    losses.append(L)

    if i % 10 == 0:
        print(f'Epoch {i}: Loss={L:.5f}, w={w:.4f}, b={b:.4f}')

    # Backpropagation — chain rule
    dL_da = -2 * (y - a)       # dL/da
    da_dz = a * (1 - a)        # da/dz — sigmoid derivative
    dL_dz = dL_da * da_dz      # dL/dz
    dL_dw = dL_dz * x          # dL/dw
    dL_db = dL_dz * 1          # dL/db

    # Weight and bias update
    w = w - lr * dL_dw
    b = b - lr * dL_db

# Final prediction
z = x * w + b
a = sigmoid(z)
print(f'\nFinal prediction: {a:.4f}')

# Loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Single Neuron Backprop — Loss Curve')
plt.tight_layout()
plt.show()