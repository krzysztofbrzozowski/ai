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
# In this case we are processing all the images through the pretrained model
# Features are extracted on the fly and the pretrained model is frozen and not trained
# The pretrained model is used as a feature extractor 
# and the top layers are trained on the extracted features (not numpy arrays as in the previous implementation)
# NO FEATURE EXTRACTION TO NUMPY ARRAYS NEEDED

# Commenting out -> on macOS runnigng tensorflow-metal is not working with keras_hub, so using keras.applications.preprocess_input() instead
# preprocessor = keras_hub.layers.ImageConverter.from_preset(
#     "xception_41_imagenet",
#     image_size=(180, 180),
# )

conv_base.trainable = False

inputs = keras.Input(shape=(180, 180, 3))

# Converts pixel values from [0, 255] to [-1, 1]
x = keras.applications.xception.preprocess_input(inputs)
# Newer preprocesing implementation commended due to issues with keras_hub on macOS
# x = preprocessor(inputs)

# Runs the pretrained model in inference mode
x = conv_base(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    loss="binary_crossentropy",
    # !IMPORTANT
    # Using a lower learning rate for the Adam optimizer to avoid large updates to the weights, for now lets use only that
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
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
# --- END FEATURE EXTRACTION FORM PRETRAINED MODEL
# --- START MODEL TRAINING

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "ch_08_convnet"
    / "dogs_vs_cats_pretrained_model_xception_head_training_only_with_data_augmentation_fine_tuning.keras"
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
    epochs=30,
    validation_data=validation_dataset,
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
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")
# Test accuracy: 0.980
# --- END MODEL EVALUATION

if __name__ == "__main__":
    conv_base.summary()
    pass