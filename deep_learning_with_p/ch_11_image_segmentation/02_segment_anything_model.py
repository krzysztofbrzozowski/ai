# --- START sam_huge_sa1b MODEL IMPORT
import sys
import types

# Monkey-patch: keras_hub 0.19.0 won't load SAM models without tensorflow-text,
# but tensorflow-text forces tf>=2.20 which breaks Metal GPU. SAM uses no text
# ops, so injecting a fake module bypasses the check safely.
_fake_tf_text = types.ModuleType("tensorflow_text")
sys.modules["tensorflow_text"] = _fake_tf_text

import keras_hub

model = keras_hub.models.ImageSegmenter.from_preset("sam_huge_sa1b")
# --- END sam_huge_sa1b MODEL IMPORT
# --- START SAMPLE IMAGE
import keras
import matplotlib.pyplot as plt

# Downloads the image and returns the local file path
path = keras.utils.get_file(
    origin="https://s3.amazonaws.com/keras.io/img/book/fruits.jpg"
)
# Loads the image as a Python Imaging Library (PIL) object
pil_image = keras.utils.load_img(path)
# Turns the PIL object into a NumPy matrix
image_array = keras.utils.img_to_array(pil_image)

# Displays the NumPy matrix
plt.imshow(image_array.astype("uint8"))
plt.axis("off")
plt.show()
# --- END SAMPLE IMAGE
# --- START PREPARING IMAGE
from keras import ops

image_size = (1024, 1024)

# Resizes the image to 1024x1024 while preserving its aspect ratio
# Any remaining space is filled with padding to avoid stretching the image
def resize_and_pad(x):
    return ops.image.resize(
        x,
        image_size,
        pad_to_aspect_ratio=True
    )

image = resize_and_pad(image_array)
# --- END PREPARING IMAGE

if __name__ == "__main__":
    pass