import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# Dataset
X, y = make_moons(100, noise=0.25, random_state=2)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Make Moons Dataset')
plt.show()

# ── No Regularization ─────────────────────────────────
model1 = Sequential([
    Dense(128, input_dim=2, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1,   activation='sigmoid')
])
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history1 = model1.fit(X, y, epochs=100, validation_split=0.2, verbose=0)
_, acc1 = model1.evaluate(X, y, verbose=0)
print(f'No Regularization — Accuracy: {acc1:.4f}')

# ── L1 Regularization ─────────────────────────────────
model2 = Sequential([
    Dense(128, input_dim=2, activation='relu',
          kernel_regularizer=regularizers.l1(0.001)),
    Dense(128, activation='relu',
          kernel_regularizer=regularizers.l1(0.001)),
    Dense(1, activation='sigmoid')
])
model2.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.01),
    metrics=['accuracy']
)
history2 = model2.fit(X, y, epochs=2000, validation_split=0.2, verbose=0)
_, acc2 = model2.evaluate(X, y, verbose=0)
print(f'L1 Regularization — Accuracy: {acc2:.4f}')

# ── L2 Regularization ─────────────────────────────────
model3 = Sequential([
    Dense(128, input_dim=2, activation='relu',
          kernel_regularizer=regularizers.l2(0.001)),
    Dense(128, activation='relu',
          kernel_regularizer=regularizers.l2(0.001)),
    Dense(1, activation='sigmoid')
])
model3.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.01),
    metrics=['accuracy']
)
history3 = model3.fit(X, y, epochs=2000, validation_split=0.2, verbose=0)
_, acc3 = model3.evaluate(X, y, verbose=0)
print(f'L2 Regularization — Accuracy: {acc3:.4f}')

# ── Compare Loss Curves ───────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

ax1.plot(history1.history['loss'],     label='train')
ax1.plot(history1.history['val_loss'], label='val')
ax1.set_title('No Regularization')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history2.history['loss'],     label='train')
ax2.plot(history2.history['val_loss'], label='val')
ax2.set_title('L1 Regularization')
ax2.set_xlabel('Epoch')
ax2.legend()

ax3.plot(history3.history['loss'],     label='train')
ax3.plot(history3.history['val_loss'], label='val')
ax3.set_title('L2 Regularization')
ax3.set_xlabel('Epoch')
ax3.legend()

plt.tight_layout()
plt.show()