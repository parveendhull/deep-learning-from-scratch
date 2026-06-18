import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- 1. ACTIVATION FUNCTIONS ---
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):

    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

# Binary Cross-Entropy Loss function
def compute_loss(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


X = np.random.randn(300, 15)
y = np.random.randint(0, 2, size=(300, 1))

# --- 3. NEURAL NETWORK TRAINING FUNCTION ---
def train_network(init_type='he', epochs=120, lr=0.05):
    # Network Layer Architecture
    n_in, n_h1, n_h2, n_out = 15, 30, 30, 1
    m = X.shape[0] # Number of samples

    # Weight Initialization Techniques Selection
    if init_type == 'zero':
        # All weights strictly 0
        W1 = np.zeros((n_in, n_h1))
        W2 = np.zeros((n_h1, n_h2))
        W3 = np.zeros((n_h2, n_out))
    elif init_type == 'random':
        # Unscaled normal distribution (0.8 scale factor)
        W1 = np.random.randn(n_in, n_h1) * 0.8
        W2 = np.random.randn(n_h1, n_h2) * 0.8
        W3 = np.random.randn(n_h2, n_out) * 0.8
    elif init_type == 'he':
        # He Formula: np.random.randn(n) * sqrt(2/n)
        W1 = np.random.randn(n_in, n_h1) * np.sqrt(2.0 / n_in)
        W2 = np.random.randn(n_h1, n_h2) * np.sqrt(2.0 / n_h1)
        W3 = np.random.randn(n_h2, n_out) * np.sqrt(2.0 / n_h2)


    b1 = np.zeros((1, n_h1))
    b2 = np.zeros((1, n_h2))
    b3 = np.zeros((1, n_out))

    loss_history = []

    # Training Loop
    for epoch in range(epochs):
        # ---- FORWARD PASS ----
        Z1 = np.dot(X, W1) + b1
        A1 = relu(Z1)

        Z2 = np.dot(A1, W2) + b2
        A2 = relu(Z2)

        Z3 = np.dot(A2, W3) + b3
        A3 = sigmoid(Z3)


        loss = compute_loss(A3, y)
        loss_history.append(loss)

        # ---- BACKWARD PASS (Backpropagation) ----
        # Layer 3 Gradients
        dZ3 = A3 - y
        dW3 = np.dot(A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # Layer 2 Gradients
        dA2 = np.dot(dZ3, W3.T)
        dZ2 = dA2 * relu_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Layer 1 Gradients
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # ---- WEIGHTS AND BIASES UPDATE ----
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3

    return loss_history

# --- 4. EXECUTION AND EVALUATION ---
print("Training models... please wait...")
losses_zero = train_network(init_type='zero')
losses_rand = train_network(init_type='random')
losses_he   = train_network(init_type='he')
print("Training complete!")

# --- 5. PLOTTING THE COMPARISON GRAPH ---
plt.figure(figsize=(10, 6))
plt.plot(losses_zero, label='Zero Initialization (All weights = 0)', color='crimson', linewidth=2.5)
plt.plot(losses_rand, label='Random Initialization (np.random.randn)', color='darkorange', linewidth=2)
plt.plot(losses_he, label='He Initialization (Scaled random)', color='forestgreen', linewidth=2)

plt.title('Weight Initialization Comparison on Same Network Topology', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.show()
