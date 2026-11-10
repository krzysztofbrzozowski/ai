import keras_hub
import numpy as np
import keras

import keras
import pathlib


import pathlib

# --- START LOADING DATASET
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
input_dir = (
    PROJECT_DIR
    / "datasets"
    / "mini-c4"
)
target_dir = (
    PROJECT_DIR
    / "datasets"
    / "mini-c4"
    / "annotations"
    / "trimaps"
)

extract_dir = keras.utils.get_file(
    fname="mini-c4",
    origin=(
        "https://hf.co/datasets/mattdangerw/mini-c4/resolve/main/mini-c4.zip"
    ),
    extract=True,
)
extract_dir = pathlib.Path(extract_dir) / "mini-c4"

vocabulary_file = keras.utils.get_file(
    origin="https://hf.co/mattdangerw/spiece/resolve/main/vocabulary.proto",
)
tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(vocabulary_file)

if __name__ == "__main__":
    pass