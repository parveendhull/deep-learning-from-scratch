import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPool2D, Flatten,
    BatchNormalization, Dropout, GlobalMaxPool2D
)

# ── Load Data ──────────────────────────────────────────
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/content/catsvsdogs/train",
    label_mode="int",
    labels="inferred",
    batch_size=32,
    image_size=(128, 128)
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/content/catsvsdogs/test",
    label_mode="int",
    labels="inferred",
    batch_size=32,
    image_size=(128, 128)
)

# ── Normalize ──────────────────────────────────────────
def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_ds = train_ds.map(process)
test_ds  = test_ds.map(process)

# ── Data Augmentation (modern Keras layers API) ───────
# Using tf.keras.layers instead of deprecated ImageDataGenerator
# Runs on GPU, integrates directly into the model graph
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# ── Model Architecture ─────────────────────────────────
model = Sequential()
model.add(data_augmentation)

model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3),
                  activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(GlobalMaxPool2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# ── Callbacks ───────────────────────────────────────────
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ── Compile and Train ───────────────────────────────────
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=30,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping_callback]
)

# ── Best Epoch Results ──────────────────────────────────
best_epoch = np.argmin(history.history['val_loss']) + 1

print("Best Epoch:", best_epoch)
print("Train Accuracy:", history.history['accuracy'][best_epoch - 1])
print("Validation Accuracy:", history.history['val_accuracy'][best_epoch - 1])
print("Train Loss:", history.history['loss'][best_epoch - 1])
print("Validation Loss:", history.history['val_loss'][best_epoch - 1])