import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load data
df = pd.read_csv('admission_data.csv')
print(df.info())
print(df.head())

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert to float32
X_train_scaled = X_train_scaled.astype('float32')
X_test_scaled  = X_test_scaled.astype('float32')
y_train        = y_train.values.astype('float32')

# Model architecture
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))  # linear for regression

model.summary()

# Compile
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# Evaluate
pred = model.predict(X_test_scaled)
r2   = r2_score(y_test, pred)
print(f'\nR² Score: {r2:.4f}')

# Plot loss curve
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'],     label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('GRE Admission — Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.tight_layout()
plt.show()