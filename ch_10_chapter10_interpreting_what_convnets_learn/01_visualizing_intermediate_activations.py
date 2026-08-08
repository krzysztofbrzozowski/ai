
# --- START MODEL LOADING
import pathlib
import keras

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "ch_08_convnet"
    / "dogs_vs_cats.keras"
)

model = keras.models.load_model(MODEL_PATH)
model.summary()
# Model: "functional"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ input_layer (InputLayer)             │ (None, 180, 180, 3)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ rescaling (Rescaling)                │ (None, 180, 180, 3)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d (Conv2D)                      │ (None, 178, 178, 32)        │             896 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d (MaxPooling2D)         │ (None, 89, 89, 32)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 87, 87, 64)          │          18,496 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_1 (MaxPooling2D)       │ (None, 43, 43, 64)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_2 (Conv2D)                    │ (None, 41, 41, 128)         │          73,856 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_2 (MaxPooling2D)       │ (None, 20, 20, 128)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_3 (Conv2D)                    │ (None, 18, 18, 256)         │         295,168 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_3 (MaxPooling2D)       │ (None, 9, 9, 256)           │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_4 (Conv2D)                    │ (None, 7, 7, 512)           │       1,180,160 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ global_average_pooling2d             │ (None, 512)                 │               0 │
# │ (GlobalAveragePooling2D)             │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 1)                   │             513 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 4,707,269 (17.96 MB)
#  Trainable params: 1,569,089 (5.99 MB)
#  Non-trainable params: 0 (0.00 B)
#  Optimizer params: 3,138,180 (11.97 MB)
# --- END MODEL LOADING
# --- LOAD IMAGE AND DISPLAY IT
import numpy as np

# Downloads a test image
img_path = keras.utils.get_file(
    fname="cat.jpg", origin="https://img-datasets.s3.amazonaws.com/cat.jpg"
)

def get_img_array(img_path, target_size):
    # Opens the image file and resizes it
    img = keras.utils.load_img(img_path, target_size=target_size)
    # Turns the image into a float32 NumPy array of shape (180, 180, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch" of a
    # single sample. Its shape is now (1, 180, 180, 3).
    array = np.expand_dims(array, axis=0)
    return array

img_tensor = get_img_array(img_path, target_size=(180, 180))

# --- DISPLAY IMAGE
import matplotlib.pyplot as plt

plt.axis("off")
plt.imshow(img_tensor[0].astype("uint8"))
plt.show()
# --- END DISPLAY IMAGE

# --- START INTERMEDIATE ACTIVATIONS
# Note:
# The model will return the outputs of all Conv2D and MaxPooling2D layers, given the model input
# This is not the same as "model", this will only return the outputs of the intermediate layers, not the final output of the model
from keras import layers

layer_outputs = []
layer_names = []
# Extracts the outputs of all Conv2D and MaxPooling2D layers and put
# them in a list
for layer in model.layers:
    if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
        layer_outputs.append(layer.output)
        # Saves the layer names for later
        layer_names.append(layer.name)
# Creates a model that will return these outputs, given the model input
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
# --- END INTERMEDIATE ACTIVATIONS
# --- START DISPLAY INTERMEDIATE ACTIVATIONS
# Returns a list of nine NumPy arrays — one array per layer activation
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
# Returns a 4D tensor with shape (1, 178, 178, 33) -> 32 channels feature maps
# (1, 178, 178, 32)

# Display channges of the first layer activation
import matplotlib.pyplot as plt

# Change last parameter to display different channels of the first layer activation
# You can use other colormaps as well, e.g. "gray", "hot", "magma", or "plasma"
plt.matshow(first_layer_activation[0, :, :, 5], cmap="viridis")

# Save all feature maps of the first layer activation to a directory
FEATURE_MAPS_DIR = (
    PROJECT_DIR
    / "docs"
    / "imgs"
    / "ch_10"
    / "feature_maps_first_layer_activation"
)

FEATURE_MAPS_DIR.mkdir(parents=True, exist_ok=True)
# Save all feature maps of the first layer activation to a directory
for i in range(first_layer_activation.shape[-1]):
    plt.matshow(first_layer_activation[0, :, :, i], cmap="viridis")
    plt.axis("off")
    plt.savefig(
        FEATURE_MAPS_DIR / f"feature_map_{i}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

# --- Save all feature maps of all layers activation to a directory
images_per_row = 16

FEATURE_MAPS_DIR = (
    PROJECT_DIR
    / "docs"
    / "imgs"
    / "ch_10"
    / "feature_maps"
)

FEATURE_MAPS_DIR.mkdir(parents=True, exist_ok=True)

for layer_no, (layer_name, layer_activation) in enumerate(
    zip(layer_names, activations)
):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]

    n_rows = n_features // images_per_row

    display_grid = np.zeros(
        (
            (size + 1) * n_rows - 1,
            images_per_row * (size + 1) - 1,
        )
    )

    for row in range(n_rows):
        for col in range(images_per_row):
            channel_index = row * images_per_row + col

            channel_image = layer_activation[
                0, :, :, channel_index
            ].copy()

            if channel_image.sum() != 0:
                channel_image -= channel_image.mean()

                # Avoid division by zero for constant feature maps
                if channel_image.std() != 0:
                    channel_image /= channel_image.std()

                channel_image *= 64
                channel_image += 128

            channel_image = np.clip(
                channel_image, 0, 255
            ).astype("uint8")

            display_grid[
                row * (size + 1) : (row + 1) * size + row,
                col * (size + 1) : (col + 1) * size + col,
            ] = channel_image

    # Saves the complete feature map grid directly as an image
    plt.imsave(
        FEATURE_MAPS_DIR
        / f"layer_no_{layer_no}_layer_name_{layer_name}.png",
        display_grid,
        cmap="viridis",
    )
# --- END DISPLAY INTERMEDIATE ACTIVATIONS

if __name__ == "__main__":
    pass