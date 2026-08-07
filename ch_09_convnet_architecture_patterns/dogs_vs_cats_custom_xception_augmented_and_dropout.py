import os, shutil, pathlib

# --- START SMALL DATASET CREATION
# Path to the directory where the original dataset was uncompressed
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent

original_dir = (
    PROJECT_DIR
    / "datasets"
    / "dogs_vs_cats"
    / "train"
).resolve()
# Directory where we will store our smaller dataset
new_base_dir = (
    PROJECT_DIR
    / "datasets"
    / "dogs_vs_cats_small"
).resolve()
# --- END SMALL DATASET CREATION
# --- START MODEL DEFINITION
import keras
from keras import layers

# Note:
# !IMPORTANT
# Custom Xception model with few features
# -> Residual connections
# -> SeparableConv2D layers
# -> BatchNormalization layers
inputs = keras.Input(shape=(180, 180, 3))
# Don't forget input rescaling!
x = layers.Rescaling(1.0 / 255)(inputs)
# The assumption that underlies separable convolution, "Feature
# channels are largely independent," does not hold for RGB images! Red,
# green, and blue color channels are actually highly correlated in
# natural images. As such, the first layer in our model is a regular
# `Conv2D` layer. We'll start using `SeparableConv2D` afterward.
x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)

# We apply a series of convolutional blocks with increasing feature
# depth. Each block consists of two batch-normalized depthwise
# separable convolution layers and a max pooling layer, with a residual
# connection around the entire block.
for size in [32, 64, 128, 256, 512]:
    residual = x

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    residual = layers.Conv2D(
        size, 1, strides=2, padding="same", use_bias=False
    )(residual)
    x = layers.add([x, residual])

# In the original model, we used a Flatten layer before the Dense
# layer. Here, we go with a GlobalAveragePooling2D layer.
x = layers.GlobalAveragePooling2D()(x)
# Like in the original model, we add a dropout layer for
# regularization.
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
# --- END MODEL DEFINITION
# --- START MODEL COMPILATION
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)
pass
# --- END MODEL COMPILATION
# --- START DATA LOADING
from keras.utils import image_dataset_from_directory

batch_size = 64
image_size = (180, 180)
train_dataset = image_dataset_from_directory(
    new_base_dir / "train", image_size=image_size, batch_size=batch_size
)
validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation", image_size=image_size, batch_size=batch_size
)
test_dataset = image_dataset_from_directory(
    new_base_dir / "test", image_size=image_size, batch_size=batch_size
)

# --- ADD DATA AUGMENTATION
import tensorflow as tf

# Defines the transformations to apply as a list
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
]

# Creates a function that applies them sequentially
def data_augmentation(images, targets):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images, targets

# Maps this function into the dataset
augmented_train_dataset = train_dataset.map(
    data_augmentation, num_parallel_calls=8
)
# Enables prefetching of batches on GPU memory; important for best
# performance
augmented_train_dataset = augmented_train_dataset.prefetch(tf.data.AUTOTUNE)
# --- END DATA LOADING
# --- START MODEL TRAINING

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "ch_09_convnet_architecture_patterns"
    / "dogs_vs_cats_custom_xception_augmented_and_dropout.keras"
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    augmented_train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=callbacks,
)
pass
# --- END MODEL TRAINING
# --- START MODEL VERIFICATION
import matplotlib.pyplot as plt

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, "r--", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()
# --- END MODEL VERIFICATION
# --- START TEST EVALUATION
test_model = keras.models.load_model(MODEL_PATH)
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")
# Test accuracy: 0.876
# --- END TEST EVALUATION

if __name__ == "__main__":
    pass