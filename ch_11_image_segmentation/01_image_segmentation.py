import pathlib

# --- START LOADING DATASET
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
input_dir = (
    PROJECT_DIR
    / "datasets"
    / "image_segmentation"
    / "images"
)
target_dir = (
    PROJECT_DIR
    / "datasets"
    / "image_segmentation"
    / "annotations"
    / "trimaps"
)
# input_dir = pathlib.Path("images")
# target_dir = pathlib.Path("annotations/trimaps")

input_img_paths = sorted(input_dir.glob("*.jpg"))
# Ignores some spurious files in the trimaps directory that start with
# a "."
target_paths = sorted(target_dir.glob("[!.]*.png"))

# --- SHOW SOME IMAGES
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array, array_to_img

plt.axis("off")
# Displays input image number 9
plt.imshow(load_img(input_img_paths[9]))
# plt.show()

def display_target(target_array):
    # The original labels are 1, 2, and 3. We subtract 1 so that the
    # labels range from 0 to 2, and then we multiply by 127 so that the
    # labels become 0 (black), 127 (gray), 254 (near-white).
    normalized_array = (target_array.astype("uint8") - 1) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])
    # plt.show()

# We use color_mode='grayscale' so that the image we load is treated as
# having a single color channel.
img = img_to_array(load_img(target_paths[9], color_mode="grayscale"))
display_target(img)

# --- PACK ALL THE DATA INTO THE NUMPY ARRAYS
import numpy as np
import random

# We resize everything to 200 x 200 for this example.
img_size = (200, 200)
# Total number of samples in the data
num_imgs = len(input_img_paths)

# Shuffles the file paths (they were originally sorted by breed). We
# use the same seed (1337) in both statements to ensure that the input
# paths and target paths stay in the same order.
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)

def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))

def path_to_target(path):
    img = img_to_array(
        load_img(path, target_size=img_size, color_mode="grayscale")
    )
    # Subtracts 1 so that our labels become 0, 1, and 2
    img = img.astype("uint8") - 1
    return img

# Loads all images in the input_imgs float32 array and their masks in
# the targets uint8 array (same order). The inputs have three channels
# (RGB values), and the targets have a single channel (which contains
# integer labels).
input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
# input_imgs = np.zeros((7390,) + (200, 200) + (3,) )
# Tensor shape: (7390, 200, 200, 3)
#
# Dimensions:
#   [0:7390] -> images
#   [0:200]  -> rows (y)
#   [0:200]  -> columns (x)
#   [0:3]    -> RGB channels
#
# input_imgs
# ├── image 0
# │   ├── row 0
# │   │   ├── pixel 0 -> [R, G, B]
# │   │   ├── pixel 1 -> [R, G, B]
# │   │   └── ...
# │   ├── row 1
# │   └── ...
# ├── image 1
# ├── image 2
# └── ...
#
# Access pattern:
# input_imgs[image, y, x, channel]
#
# Examples:
# input_imgs[0]             -> shape (200, 200, 3)
# input_imgs[0, 0]          -> shape (200, 3)
# input_imgs[0, 0, 0]       -> shape (3,) = [R, G, B]
# input_imgs[0, 0, 0, 0]    -> scalar (red channel value)
# ---
# Shape: (7390, 200, 200, 3)
#        |     |    |    |
#        |     |    |    +-- RGB channel [0:3]
#        |     |    +------- column (x) [0:200]
#        |     +------------ row (y) [0:200]
#        +------------------ image index [0:7390]
#
# Access: input_imgs[image, y, x, channel]
targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")
for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i])

# --- VALIDATION AND TRAINING DATASET
# Reserves 1,000 samples for validation
num_val_samples = 1000
# Splits the data into a training and a validation set
# train_input_imgs = input_imgs[0:7390-1000] -> input_imgs[0:6390] (first 6390, 1000 left)
train_input_imgs = input_imgs[:-num_val_samples]
# train_targets = targets[0:7390-1000] -> targets[0:6390] (first 6390, 1000 left)
train_targets = targets[:-num_val_samples]
# interpret as -> 7390-1000
# val_input_imgs = input_imgs[7390-1000:7390] -> input_imgs[6390:7390] (last 1000)
val_input_imgs = input_imgs[-num_val_samples:]
# val_targets = targets[7390-1000:7390] -> targets[6390:7390] (last 1000)
val_targets = targets[-num_val_samples:]
# --- END LOADING DATASET
# --- START BUILDING THE MODEL
import keras
from keras.layers import Rescaling, Conv2D, Conv2DTranspose

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    # Don't forget to rescale input images to the [0–1] range.
    x = Rescaling(1.0 / 255)(inputs)

    # We use padding="same" everywhere to avoid the influence of border
    # padding on feature map size.
    x = Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    x = Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2D(128, 3, activation="relu", padding="same")(x)
    x = Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = Conv2D(256, 3, activation="relu", padding="same")(x)

    x = Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(256, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)

    # We end the model with a per-pixel three-way softmax to classify
    # each output pixel into one of our three categories.
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    return keras.Model(inputs, outputs)

model = get_model(img_size=img_size, num_classes=3)
# --- END BUILDING THE MODEL
# --- START COMPILING AND TRAINING THE MODEL
foreground_iou = keras.metrics.IoU(
    # Specifies the total number of classes
    num_classes=3,
    # Specifies the class to compute IoU for (0 = foreground)
    target_class_ids=(0,),
    name="foreground_iou",
    # Our targets are sparse (integer class IDs).
    sparse_y_true=True,
    # But our model's predictions are a dense softmax!
    sparse_y_pred=False,
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[foreground_iou],
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "ch_11_image_segmentation"
    / "oxford_segmentation.keras"
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True,
    ),
]
history = model.fit(
    train_input_imgs,
    train_targets,
    epochs=50,
    callbacks=callbacks,
    batch_size=64,
    validation_data=(val_input_imgs, val_targets),
)
# --- END COMPILING AND TRAINING THE MODEL
# --- START MODEL LOSS AND METRICS PLOT
# Loss
epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.figure()
plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()

# IoU
epochs = range(1, len(history.history["foreground_iou"]) + 1)
iou = history.history["foreground_iou"]
val_iou = history.history["val_foreground_iou"]
plt.figure()
plt.plot(epochs, iou, "r--", label="Training IoU")
plt.plot(epochs, val_iou, "b", label="Validation IoU")
plt.title("Training and validation IoU")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.legend()
# --- END MODEL LOSS AND METRICS PLOT
# --- START DISPLAYING SOME PREDICTIONS
model = keras.models.load_model("oxford_segmentation.keras")

i = 1337
test_image = val_input_imgs[i]
plt.axis("off")
plt.imshow(array_to_img(test_image))

mask = model.predict(np.expand_dims(test_image, 0))[0]

# Utility to display a model's prediction
def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)

display_mask(mask)
# --- END DISPLAYING SOME PREDICTIONS


if __name__ == "__main__":
    pass