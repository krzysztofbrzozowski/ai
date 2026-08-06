# --- START PRETRAINED MODEL
# Commenting out -> on macOS runnigng tensorflow-metal is not working with keras_hub, so using keras.applications instead
# import keras_hub
#
# conv_base = keras_hub.models.Backbone.from_preset("xception_41_imagenet")
#
# Old implementation
from tensorflow import keras

conv_base = keras.applications.Xception(weights="imagenet", include_top=False)
# --- END PRETRAINED MODEL
# --- START MODEL DEFINITION
from keras import layers
# Note:
# !IMPORTANT
# In this case we are applying only top layers to the features extracted 
# from the pretrained model -> features are extracted to the numpy arrays
# The pretrained model is frozen and not trained
inputs = keras.Input(shape=(6, 6, 2048))
# Averages spatial dimensions to flatten the feature map
x = layers.GlobalAveragePooling2D()(inputs)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)
# --- END MODEL DEFINITION
# --- START DATA LOADING
import os, shutil, pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent

# Directory where we will store our smaller dataset
new_base_dir = (
    PROJECT_DIR
    / "datasets"
    / "dogs_vs_cats_small"
).resolve()

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
# --- END DATA LOADING
# --- START FEATURE EXTRACTION
# Commenting out -> on macOS runnigng tensorflow-metal is not working with keras_hub, so using keras.applications.preprocess_input() instead
# preprocessor = keras_hub.layers.ImageConverter.from_preset(
#     "xception_41_imagenet",
#     image_size=(180, 180),
# )

import numpy as np
def get_features_and_labels(dataset):
    all_features = []
    all_labels = []

    for images, labels in dataset:
        preprocessed_images = (
            # Use older implementation of preprocessor to avoid issues with keras_hub on macOS
            keras.applications.xception.preprocess_input(images)
        )
        # Commenting out -> on macOS runnigng tensorflow-metal is not working with keras_hub, so using keras.applications.preprocess_input() instead
        # preprocessed_images = preprocessor(images)

        # Main part: extract features using the pretrained model
        features = conv_base.predict(
            preprocessed_images,
            verbose=0,
        )

        all_features.append(features)
        all_labels.append(labels)

    return (
        np.concatenate(all_features),
        np.concatenate(all_labels),
    )

# EXTRTACT FEATURES FOR ALL DATASETS FORM PRETRAINED MODEL
train_features, train_labels = get_features_and_labels(train_dataset)
val_features, val_labels = get_features_and_labels(validation_dataset)
test_features, test_labels = get_features_and_labels(test_dataset)
# --- END FEATURE EXTRACTION FORM PRETRAINED MODEL
# --- START MODEL TRAINING

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "ch_08_convnet"
    / "dogs_vs_cats_pretrained_model_xception_head_training_only.keras"
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    train_features,
    train_labels,
    epochs=10,
    validation_data=(val_features, val_labels),
    callbacks=callbacks,
)
# --- END MODEL TRAINING
# --- START MODEL VERIFICATION
import matplotlib.pyplot as plt

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "r--", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()
# --- END MODEL VERIFICATION
# --- START MODEL EVALUATION
test_model = keras.models.load_model(MODEL_PATH)
test_loss, test_acc = test_model.evaluate(test_features, test_labels)
print(f"Test accuracy: {test_acc:.3f}")
# Test accuracy: 0.983
# --- END MODEL EVALUATION

if __name__ == "__main__":
    conv_base.summary()
    pass