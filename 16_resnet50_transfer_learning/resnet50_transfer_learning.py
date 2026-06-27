import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Load Data ──────────────────────────────────────────
# Note: 224x224 used — ResNet50's native input size.
# Using a smaller size (e.g. 128x128) caused unstable
# training (see README for full debugging story).
IMG_SIZE = (224, 224)

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/content/catsvsdogs/train",
    label_mode="int",
    labels="inferred",
    batch_size=32,
    image_size=IMG_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/content/catsvsdogs/test",
    label_mode="int",
    labels="inferred",
    batch_size=32,
    image_size=IMG_SIZE
)

def process(image, label):
    image = tf.cast(image, tf.float32)  # no /255 — preprocess_input handles scaling
    return image, label

train_ds = train_ds.map(process)
test_ds  = test_ds.map(process)

# ── Data Augmentation ──────────────────────────────────
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(*IMG_SIZE, 3)),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# ── Build Model — Feature Extraction Phase ────────────
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False  # freeze pretrained weights

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)            # ResNet50-specific normalization
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

# ── Phase 1: Train top layer only ─────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

EPOCHS = 5
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

train_acc = history.history['accuracy'][-1]
val_acc   = history.history['val_accuracy'][-1]
print(f"Feature Extraction — Train Accuracy: {train_acc*100:.2f}%, Val Accuracy: {val_acc*100:.2f}%")

# ── Phase 2: Fine-tune last 30 layers ─────────────────
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # much smaller LR
    loss='binary_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 5
total_epochs = EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1]
)

fine_train_acc = history_fine.history['accuracy'][-1]
fine_val_acc   = history_fine.history['val_accuracy'][-1]
print(f"Fine-Tuning — Train Accuracy: {fine_train_acc*100:.2f}%, Val Accuracy: {fine_val_acc*100:.2f}%")