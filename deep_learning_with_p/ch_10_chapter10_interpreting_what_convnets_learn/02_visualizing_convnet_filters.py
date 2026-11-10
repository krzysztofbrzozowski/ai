from tensorflow import keras

# --- START DOWNLOAD AND PREPROCESS TEST IMAGE
import numpy as np
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
# --- END DOWNLOAD AND PREPROCESS TEST IMAGE
# --- START LOAD PRETRAINED MODEL AND PREPROCESSOR
conv_base = keras.applications.Xception(weights="imagenet", include_top=False)

# Converts pixel values from [0, 255] to [-1, 1]
preprocessor = keras.applications.xception.preprocess_input
# Newer preprocesing implementation commended due to issues with keras_hub on macOS
# x = preprocessor(inputs)
# --- END LOAD PRETRAINED MODEL AND PREPROCESSOR


for layer in conv_base.layers:
    if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
        print(layer.name)
# block1_conv1
# block1_conv2
# block2_sepconv1
# block2_sepconv2
# conv2d
# block3_sepconv1
# block3_sepconv2
# conv2d_1
# block4_sepconv1
# block4_sepconv2
# conv2d_2
# block5_sepconv1
# block5_sepconv2
# block5_sepconv3
# block6_sepconv1
# block6_sepconv2
# block6_sepconv3
# block7_sepconv1
# block7_sepconv2
# block7_sepconv3
# block8_sepconv1
# block8_sepconv2
# block8_sepconv3
# block9_sepconv1
# block9_sepconv2
# block9_sepconv3
# block10_sepconv1
# block10_sepconv2
# block10_sepconv3
# block11_sepconv1
# block11_sepconv2
# block11_sepconv3
# block12_sepconv1
# block12_sepconv2
# block12_sepconv3
# block13_sepconv1
# block13_sepconv2
# conv2d_3
# block14_sepconv1
# block14_sepconv2
# You could replace this with the name of any layer in the Xception
# convolutional base.

# Cut the modedel at the layer we want to visualize
layer_name = "block3_sepconv1"
# This is the layer object we're interested in.
layer = conv_base.get_layer(name=layer_name)
# We use conv_base.input and layer.output to create a model that, given an
# input image, returns the output of our target layer.
feature_extractor = keras.Model(inputs=conv_base.input, outputs=layer.output)


activation = feature_extractor(preprocessor(img_tensor))

from keras import ops

# The loss function takes an image tensor and the index of the filter
# we consider (an integer).
# This function measures how strongly the current image activates the selected filter
# This gives us a score, but does not tell us how to change the image (will be done by gradient ascent)
# Gradient ascent will use dLoss/dImage to modify the pixels
# So that the selected filter becomes more and more activated
# The final image shows the visual pattern that the filter responds to most strongly
def compute_loss(image, filter_index):
    # Note:
    # Calling model(image) vs model.predict(image) is different
    # model(image) allows us to compute gradients, while model.predict(image) does not
    # Here we are calculating the loss for specific filter based on activation by the image
    # Normally we are calculating we calculate gradients dLoss/dW and using in gradient_descent calculation
    # Here we are calculating dLoss/dImage ->
    # This will allow us to determine which pixels in the image are most responsible for activating the filter
    activation = feature_extractor(image)
    # We avoid border artifacts by only involving nonborder pixels in
    # the loss: we discard the first 2 pixels along the sides of the
    # activation.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    # Returns the mean of the activation values for the filter
    return ops.mean(filter_activation)

import tensorflow as tf


@tf.function
def gradient_ascent_step(image, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        # Explicitly watches the image tensor, since it isn't a
        # TensorFlow Variable (only Variables are automatically watched
        # in a gradient tape)
        tape.watch(image)
        # Computes the loss scalar, indicating how much the current
        # image activates the filter
        loss = compute_loss(image, filter_index)
    # Computes the gradients of the loss with respect to the image
    grads = tape.gradient(loss, image)
    # Applies the "gradient normalization trick"
    grads = ops.normalize(grads)
    # Moves the image a little bit in a direction that activates our
    # target filter more strongly
    image += learning_rate * grads
    # Returns the updated image, so we can run the step function in a
    # loop
    return image

# --- START GENERATE FILTER PATTERNS
img_width = 200
img_height = 200

def generate_filter_pattern(filter_index):
    # The number of gradient ascent steps to apply
    iterations = 30
    # The amplitude of a single step
    learning_rate = 10.0
    image = keras.random.uniform(
        # Initialize an image tensor with random values. (The Xception
        # model expects input values in the [0, 1] range, so here we
        # pick a range centered on 0.5.)
        minval=0.4, maxval=0.6, shape=(1, img_width, img_height, 3)
    )
    # Repeatedly updates the values of the image tensor to maximize our
    # loss function
    for i in range(iterations):
        image = gradient_ascent_step(image, filter_index, learning_rate)
    return image[0]


def deprocess_image(image):
    # Normalizes image values within the [0, 255] range
    image -= ops.mean(image)
    image /= ops.std(image)
    image *= 64
    image += 128
    image = ops.clip(image, 0, 255)
    # Center crop to avoid border artifacts
    image = image[25:-25, 25:-25, :]
    image = ops.cast(image, dtype="uint8")
    return ops.convert_to_numpy(image)


plt.axis("off")
plt.imshow(deprocess_image(generate_filter_pattern(filter_index=2)))

# Generates and saves visualizations for the first 64 filters in the layer
all_images = []
for filter_index in range(64):
    print(f"Processing filter {filter_index}")
    image = deprocess_image(generate_filter_pattern(filter_index))
    all_images.append(image)

# Prepares a blank canvas for us to paste filter visualizations
margin = 5
n = 8
box_width = img_width - 25 * 2
box_height = img_height - 25 * 2
full_width = n * box_width + (n - 1) * margin
full_height = n * box_height + (n - 1) * margin
stitched_filters = np.zeros((full_width, full_height, 3))

# Fills the picture with our saved filters
for i in range(n):
    for j in range(n):
        image = all_images[i * n + j]
        stitched_filters[
            (box_width + margin) * i : (box_width + margin) * i + box_width,
            (box_height + margin) * j : (box_height + margin) * j + box_height,
            :,
        ] = image

import pathlib
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
FEATURE_MAPS_DIR = (
    PROJECT_DIR
    / "docs"
    / "imgs"
    / "ch_10"
    / "filters_reaction_xception"
)

# Saves the canvas to disk
keras.utils.save_img(f"{FEATURE_MAPS_DIR}/filters_for_layer_{layer_name}.png", stitched_filters)

if __name__ == "__main__":
    conv_base.summary()