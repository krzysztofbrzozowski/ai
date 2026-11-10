import pathlib
import keras
import keras_hub
import numpy as np

# --- START LOADING DATASET
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent

extract_dir = keras.utils.get_file(
    fname="mini-c4",
    origin=(
        "https://hf.co/datasets/mattdangerw/mini-c4/resolve/main/mini-c4.zip"
    ),
    extract=True,
    cache_dir=PROJECT_DIR,
)
extract_dir = pathlib.Path(extract_dir) / "mini-c4"

vocabulary_file = keras.utils.get_file(
    origin="https://hf.co/mattdangerw/spiece/resolve/main/vocabulary.proto",
    cache_dir=PROJECT_DIR,
)
tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(vocabulary_file)

# Example usage of tokenizer
# >>> tokenizer.tokenize("The quick brown fox.")
# array([  450,  4996, 17354,  1701, 29916, 29889], dtype=int32)
# >>> tokenizer.detokenize([450, 4996, 17354, 1701, 29916, 29889])
# "The quick brown fox."


if __name__ == "__main__":
    pass