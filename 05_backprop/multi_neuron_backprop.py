import numpy as np
import matplotlib.pyplot as plt

# Inputs and target
x1 = 2
x2 = 3
y  = 1

# Learning rate
lr = 0.1

# Hidden layer weights and biases (2 neurons)
w1, w2, b1 = 0.5, 0.3, 1.0   # neuron 1
w3, w4, b2 = 0.2, 0.4, 1.0   # neuron 2

# Output layer weights and bias (1 neuron)
w5, w6, b3 = 0.6, 0.7, 1.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

losses = []

for i in range(1001):
    # Forward pass
    z1 = x1*w1 + x2*w2 + b1
    z2 = x1*w3 + x2*w4 + b2
    a1 = sigmoid(z1)
    a2 = sigmoid(z2)

    z3 = a1*w5 + a2*w6 + b3
    a3 = sigmoid(z3)

    L = (y - a3) ** 2
    losses.append(L)

    if i % 200 == 0:
        print(f'Epoch {i}: Loss={L:.5f}')

    # Backpropagation — chain rule
    dL_da3  = -2 * (y - a3)
    da3_dz3 = a3 * (1 - a3)
    da1_dz1 = a1 * (1 - a1)
    da2_dz2 = a2 * (1 - a2)

    # Output layer gradients
    dL_dz3  = dL_da3 * da3_dz3
    dL_dw5  = dL_dz3 * a1
    dL_dw6  = dL_dz3 * a2
    dL_db3  = dL_dz3

    # Hidden layer gradients — chain rule back through output
    dL_dz1  = dL_dz3 * w5 * da1_dz1
    dL_dz2  = dL_dz3 * w6 * da2_dz2

    dL_dw1  = dL_dz1 * x1
    dL_dw2  = dL_dz1 * x2
    dL_dw3  = dL_dz2 * x1
    dL_dw4  = dL_dz2 * x2
    dL_db1  = dL_dz1
    dL_db2  = dL_dz2

    # Update all weights and biases
    w1 = w1 - lr * dL_dw1
    w2 = w2 - lr * dL_dw2
    w3 = w3 - lr * dL_dw3
    w4 = w4 - lr * dL_dw4
    w5 = w5 - lr * dL_dw5
    w6 = w6 - lr * dL_dw6
    b1 = b1 - lr * dL_db1
    b2 = b2 - lr * dL_db2
    b3 = b3 - lr * dL_db3

# Loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Multi Neuron Backprop — Loss Curve')
plt.tight_layout()
plt.show()