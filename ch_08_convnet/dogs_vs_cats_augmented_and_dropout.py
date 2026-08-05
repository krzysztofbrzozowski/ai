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

# Uncomment the following lines to create a smaller dataset for experimentation
# # Utility function to copy cat (respectively, dog) images from index
# # `start_index` to index `end_index` to the subdirectory
# # `new_base_dir/{subset_name}/cat` (respectively, dog). "subset_name"
# # will be either "train," "validation," or "test."
# def make_subset(subset_name, start_index, end_index):
#     for category in ("cat", "dog"):
#         dir = new_base_dir / subset_name / category
#         os.makedirs(dir)
#         fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
#         for fname in fnames:
#             shutil.copyfile(src=original_dir / fname, dst=dir / fname)

# # Creates the training subset with the first 1,000 images of each
# # category
# make_subset("train", start_index=0, end_index=1000)
# # Creates the validation subset with the next 500 images of each
# # category
# make_subset("validation", start_index=1000, end_index=1500)
# # Creates the test subset with the next 1,000 images of each category
# make_subset("test", start_index=1500, end_index=2500)
# --- END SMALL DATASET CREATION
# --- START MODEL DEFINITION
import keras
from keras import layers

# The depth of the feature maps progressively increases from 32 to 512,
# while their spatial dimensions progressively decrease from 180 x 180
# to 7 x 7, which is a typical pattern in convolutional neural networks
#
# The 7 x 7 size comes from alternating valid 3 x 3 convolutions and
# 2 x 2 max-pooling operations:
# 180 -> 178 -> 89 -> 87 -> 43 -> 41 -> 20 -> 18 -> 9 -> 7

# The model expects RGB images of size 180 x 180
inputs = keras.Input(shape=(180, 180, 3))

# Rescales inputs to the [0, 1] range by dividing them by 255
x = layers.Rescaling(1.0 / 255)(inputs)

x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)      # 180 -> 178
x = layers.MaxPooling2D(pool_size=2)(x)                                 # 178 -> 89

x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)      # 89 -> 87
x = layers.MaxPooling2D(pool_size=2)(x)                                 # 87 -> 43

x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)     # 43 -> 41
x = layers.MaxPooling2D(pool_size=2)(x)                                 # 41 -> 20

x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)     # 20 -> 18
x = layers.MaxPooling2D(pool_size=2)(x)                                 # 18 -> 9

x = layers.Conv2D(filters=512, kernel_size=3, activation="relu")(x)     # 9 -> 7

# Converts the 3D activations with shape (7, 7, 512) into a 1D vector
# with shape (512,) by averaging each feature map over its spatial dimensions
x = layers.GlobalAveragePooling2D()(x)
# Dropout is a regularization technique that randomly sets a fraction of the input units to 0 at each update during training time, which helps prevent overfitting
x = layers.Dropout(0.25)(x)

# This is a binary classification problem with two possible classes
# The output layer contains one neuron
# Sigmoid converts the output value into a probability between 0 and 1
# The probability represents the predicted likelihood of class 1
# Values below 0.5 are usually classified as class 0
# Values equal to or above 0.5 are usually classified as class 1
# Binary crossentropy compares this probability with the true label 0 or 1
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

# --- DISPLAY AUGMENTED IMAGES
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
# You can use take(N) to only sample N batches from the dataset. This
# is equivalent to inserting a break in the loop after the Nth batch.
for image_batch, _ in train_dataset.take(1):
    image = image_batch[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image, _ = data_augmentation(image, None)
        augmented_image = keras.ops.convert_to_numpy(augmented_image)
        # Displays the first image in the output batch. For each of the
        # nine iterations, this is a different augmentation of the same
        # image.
        plt.imshow(augmented_image.astype("uint8"))
        plt.axis("off")
pass
# --- END DATA LOADING
# --- START MODEL TRAINING

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "ch_08_convnet"
    / "dogs_vs_cats_augmented_and_dropout.keras"
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
# Test accuracy: 0.825
# --- END TEST EVALUATION

if __name__ == "__main__":
    pass