import numpy as np
from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data[:100,:2]
y = iris.target[:100]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr= learning_rate
        self.epochs = epochs


    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()
        # Training Loop
        for epoch in range(self.epochs):
            for idx,x_i in enumerate(X):
                linear_output = np.dot(x_i,self.weights) + self.bias
                y_pred=self.activation(linear_output)
            # Perceptron update rule
                error = y[idx] - y_pred
                self.weights += self.lr * error * x_i
                self.bias += self.lr * error

    def activation(self, x):
        return 1 if x >= 0 else 0
    def predict(self, X):
        linear_output = np.dot(X,self.weights) + self.bias
        # predictions = []

        return np.where(linear_output >= 0, 1, 0)


#train model

model = Perceptron()
model.fit(X_train, y_train)

#Predictions
predictions = model.predict(X_test)
print(predictions)
# Accuracy
accuracy = sum(predictions == y_test) / len(y_test)

print("Accuracy:", accuracy)
print("Weights:", model.weights)
print("Bias:", model.bias)

