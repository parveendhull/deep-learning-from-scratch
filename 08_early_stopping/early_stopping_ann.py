import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load Dataset
df = pd.read_csv("admission_data.csv")

# Features and Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Scaling
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Early Stopping Callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Model
model = Sequential([
    Dense(8, activation='relu', input_shape=(7,)),
    Dense(1, activation='linear')
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Training
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=500,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluation
y_pred = model.predict(X_test_scaled)

print("R2 Score:", r2_score(y_test, y_pred))