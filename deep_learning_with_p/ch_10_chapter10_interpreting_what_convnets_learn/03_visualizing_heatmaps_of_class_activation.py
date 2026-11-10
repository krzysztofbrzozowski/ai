# --- START GETTING THE IMAGE
import numpy as np
import keras
# Downloads the image and stores it locally under the path img_path
img_path = keras.utils.get_file(
    fname="elephant.jpg",
    origin="https://img-datasets.s3.amazonaws.com/elephant.jpg",
)
# Returns a Python Imaging Library (PIL) image
img = keras.utils.load_img(
    img_path,
    target_size=(299, 299)
)
# Adds a batch dimension at axis 0: (height, width, channels) -> (1, height, width, channels)
img_array = np.expand_dims(img, axis=0)
# --- END GETTING THE IMAGE
# --- START LOADING THE MODEL
model = keras.applications.Xception(
    weights="imagenet",
    include_top=True,
)

img_array = keras.applications.xception.preprocess_input(img_array)
# --- END LOADING THE MODEL
# --- START TEST PREDICTING THE CLASS OF THE IMAGE
preds = model.predict(img_array)
# Decodes the predictions into a list of tuples (class, description, probability)
decoded = keras.applications.xception.decode_predictions(
    preds,
    top=10,
)

# preds contains 1000 prediction scores, one for each ImageNet class
# The highest score is at index 386, which corresponds to the "African elephant" class
np.argmax(preds[0])
# 386
# --- END TEST PREDICTING THE CLASS OF THE IMAGE
# --- START PREPARING THE MODEL FOR GRAD-CAM
# The last convolutional layer of the model is the one we want to visualize
last_conv_layer_name = "block14_sepconv2_act"
last_conv_layer = model.get_layer(last_conv_layer_name)

last_conv_layer_model = keras.Model(
    inputs=model.input,
    outputs=last_conv_layer.output,
)

# The classifier model takes the output of the last convolutional layer as input and
# returns the final predictions
classifier_input = last_conv_layer.output

x = model.get_layer("avg_pool")(classifier_input)
x = model.get_layer("predictions")(x)

classifier_model = keras.Model(
    inputs=classifier_input,
    outputs=x,
)

# Originally in the book was:
# classifier_input = last_conv_layer.output
# x = classifier_input
# for layer_name in ["pooler", "predictions"]:
#     x = model.get_layer(layer_name)(x)
# classifier_model = keras.Model(classifier_input, x)

# --- END PREPARING THE MODEL FOR GRAD-CAM
# --- START GETTING THE GRADIENTS OF THE TOP PREDICTED CLASS
import tensorflow as tf
from keras import ops

def get_top_class_gradients(img_array):
    # Computes activations of the last conv layer and makes the tape
    # watch it
    last_conv_layer_output = last_conv_layer_model(img_array)
    with tf.GradientTape() as tape:
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = ops.argmax(preds[0])
        # Retrieves the activation channel corresponding to the top
        # predicted class
        top_class_channel = preds[:, top_pred_index]

    # Gets the gradient of the top predicted class with regard to the
    # output feature map of the last convolutional layer
    # ---
    # Computes the gradient of the top predicted class score with respect 
    # to the output of the last convolutional layer
    # For example, if the top class is "African elephant", this computes:
    # dScore(African elephant) / dLastConvFeatureMaps
    # The gradient tells us how much each activation in the last conv feature maps
    # influences the score of the "African elephant" class
    #
    # A Grad-CAM shows which spatial regions of the input image were important for a specific class
    #
    # The process is:
    # feature maps
    # + gradients showing how important their activations are for the selected class
    # -> weighted feature maps
    # -> summation
    # -> spatial heatmap
    #
    # Example:
    # F1 * w1 = feature map 1 * 0.01   # very low importance for the selected class
    # F2 * w2 = feature map 2 * 0.80   # very high importance for the selected class
    # F3 * w3 = feature map 3 * 0.25   # moderate importance for the selected class
    #
    # Then all weighted feature maps are summed:
    # F1*w1 + F2*w2 + F3*w3 + ... -> final Grad-CAM heatmap
    #
    # A larger weight means that the corresponding feature map is more important
    # for the selected class prediction
    #
    # For example, for the "African elephant" class, the final heatmap highlights
    # the image regions that contributed the most to the "African elephant" prediction
    #
    # The gradients themselves tell us how important the feature map activations
    # are for the selected class
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    return grads, last_conv_layer_output

grads, last_conv_layer_output = get_top_class_gradients(img_array)
grads = ops.convert_to_numpy(grads)
last_conv_layer_output = ops.convert_to_numpy(last_conv_layer_output)
# --- END GETTING THE GRADIENTS OF THE TOP PREDICTED CLASS
# --- START DISPLAYING THE GRAD-CAM HEATMAP
# This is a vector where each entry is the mean intensity of the
# gradient for a given channel. It quantifies the importance of each
# channel with regard to the top predicted class.
import matplotlib.pyplot as plt
pooled_grads = np.mean(grads, axis=(0, 1, 2))
last_conv_layer_output = last_conv_layer_output[0].copy()
# Multiplies each channel in the output of the last convolutional layer
# by how important this channel is
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]
# The channel-wise mean of the resulting feature map is our heatmap of
# class activation.
heatmap = np.mean(last_conv_layer_output, axis=-1)

import pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
HEAT_MAPS_DIR = (
    PROJECT_DIR
    / "docs"
    / "imgs"
    / "ch_10"
    / "heat_maps"
)
HEAT_MAPS_DIR.mkdir(parents=True, exist_ok=True)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

plt.matshow(heatmap)
plt.axis("off")
plt.savefig(
    HEAT_MAPS_DIR / "heatmap.png",
    bbox_inches="tight",
    pad_inches=0,
)
plt.close()

# ---
import matplotlib
import matplotlib.pyplot as plt

# Loads the original image
img = keras.utils.load_img(img_path)
img = keras.utils.img_to_array(img)

# Rescales the heatmap to the range 0-255
heatmap = np.uint8(255 * heatmap)

# Uses the "jet" colormap to recolorize the heatmap
jet = matplotlib.colormaps["jet"]
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# Creates an image that contains the recolorized heatmap
jet_heatmap = keras.utils.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.utils.img_to_array(jet_heatmap)

# Superimposes the heatmap and the original image, with the heatmap at 40% opacity
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.utils.array_to_img(superimposed_img)

# Saves the superimposed image
superimposed_img.save(
    HEAT_MAPS_DIR / "superimposed_heatmap.png"
)

# Shows the superimposed image
plt.imshow(superimposed_img)
plt.axis("off")
plt.show()
if __name__ == "__main__":
    pass