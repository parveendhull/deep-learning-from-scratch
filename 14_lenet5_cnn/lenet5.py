import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten, AveragePooling2D
from tensorflow.keras import Sequential

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize first, then pad (28x28 -> 32x32, original LeNet input size)
X_train = tf.cast(X_train, tf.float32) / 255.0
X_test  = tf.cast(X_test, tf.float32)  / 255.0

X_train = tf.pad(X_train, [[0, 0], [2, 2], [2, 2]])
X_test  = tf.pad(X_test,  [[0, 0], [2, 2], [2, 2]])

# Add channel dimension
X_train = tf.expand_dims(X_train, axis=-1)
X_test  = tf.expand_dims(X_test,  axis=-1)

# LeNet-5 Architecture (LeCun et al., 1998)
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), activation='tanh',
                  input_shape=(32, 32, 1), padding='valid'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(16, kernel_size=(5, 5), activation='tanh', padding='valid'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84,  activation='tanh'))
model.add(Dense(10,  activation='softmax'))

model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test)
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")