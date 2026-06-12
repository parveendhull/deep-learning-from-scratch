import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data
df = pd.read_csv('admission_data.csv')
X  = df.iloc[:, :-1]
y  = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Without Scaling
model1 = Sequential([
    Dense(8, activation='relu', input_shape=(7,)),
    Dense(1, activation='linear')
])
model1.compile(optimizer='adam', loss='mse', metrics=['mae'])
model1.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)
y_pred1 = model1.predict(X_test)
print(f'Without Scaling R²: {r2_score(y_test, y_pred1):.4f}')

# With Scaling
scaler      = MinMaxScaler()
X_train_sc  = scaler.fit_transform(X_train).astype('float32')
X_test_sc   = scaler.transform(X_test).astype('float32')
y_train_sc  = y_train.values.astype('float32')

model2 = Sequential([
    Dense(8, activation='relu', input_shape=(7,)),
    Dense(1, activation='linear')
])
model2.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model2.fit(
    X_train_sc, y_train_sc,
    epochs=100,
    validation_split=0.2,
    verbose=0
)
y_pred2 = model2.predict(X_test_sc)
print(f'With Scaling    R²: {r2_score(y_test, y_pred2):.4f}')

# Loss curve
plt.plot(history.history['loss'],     label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Feature Scaling — Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.show()