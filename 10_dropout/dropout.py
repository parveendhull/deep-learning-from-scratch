import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Dataset
X, y = make_circles(n_samples=220, noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Without Dropout
model1 = Sequential([
    Dense(128, input_dim=2, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1,   activation='sigmoid')
])
model1.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)
history1 = model1.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    verbose=0
)
loss1, acc1 = model1.evaluate(X_test, y_test, verbose=0)
print(f'Without Dropout — Accuracy: {acc1:.4f}')

# With Dropout
model2 = Sequential([
    Dense(128, input_dim=2, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1,   activation='sigmoid')
])
model2.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.01),
    metrics=['accuracy']
)
history2 = model2.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    verbose=0
)
loss2, acc2 = model2.evaluate(X_test, y_test, verbose=0)
print(f'With Dropout    — Accuracy: {acc2:.4f}')

# Compare plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history1.history['loss'],     label='train')
ax1.plot(history1.history['val_loss'], label='val')
ax1.set_title('Without Dropout')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history2.history['loss'],     label='train')
ax2.plot(history2.history['val_loss'], label='val')
ax2.set_title('With Dropout (0.5)')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.show()