from keras.datasets import imdb

# --- START SAMPLE DATASET
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000
)
# Decode the reviews back to words
# word_index is a dictionary mapping words to an integer index.
word_index = imdb.get_word_index()
# Reverses it, mapping integer indices to words
reverse_word_index = dict([
    (value, key) for 
    (key, value) in word_index.items()])
# Decodes the review. Note that the indices are offset by 3 because 0,
# 1, and 2 are reserved indices for "padding," "start of sequence," and
# "unknown."
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]]
)
# --- END SAMPLE DATASET

# --- START DATA PREPROCESSING
import numpy as np

def multi_hot_encode(sequences, num_classes):
    # Example input:
    # sequences = [
    #     [2, 5],
    #     [1, 2, 7]
    # ]
    #
    # Initial results matrix:
    # [
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0]
    # ]
    #
    # After multi-hot encoding:
    # [
    #     [0, 0, 1, 0, 0, 1, 0, 0],
    #     [0, 1, 1, 0, 0, 0, 0, 1]
    # ]
    # Creates an all-zero matrix of shape (len(sequences), num_classes)
    results = np.zeros((len(sequences), num_classes))
    for i, sequence in enumerate(sequences):
        # Sets specific indices of results[i] to 1s
        # -> Function called advanced indexing
        results[i][sequence] = 1.0
    return results

# Vectorized training data
x_train = multi_hot_encode(train_data, num_classes=10000)
# Vectorized test data
x_test = multi_hot_encode(test_data, num_classes=10000)

y_train = train_labels.astype("float32")
y_test = test_labels.astype("float32")
# --- END DATA PREPROCESSING

# --- START MODEL DEFINITION
# Note: The input data is vectors, and the labels are scalars (1s and 0s) 
# -> A type of model that performs well on such a problem
# is a plain stack of densely connected (Dense) layers with relu activations
# ---
# There are two key architecture decisions to be made about such a stack of Dense layers:
# -> How many layers to use?
# -> How many units to choose for each layer?
# Note: Don't know why but suggestion is to use Dense16 relu, Dense16 relu and Dense1 sigmoid
#

import keras
from keras import layers

# Dense(16): 10000 inputs → 16 outputs
# Dense(16): 16 inputs → 16 outputs
# Dense(1): 16 inputs → 1 output
#
# Input -> (1, 10000), W -> (10000, 16), b -> (1, 16)
# Note: 
# Input @ W -> (1, 16) <-- project the input data onto a 16-dimensional representation space
# dimensionality (16 in this case) -> “how much freedom you’re allowing 
# the model to have when learning internal representations.”
#
# Note:
# A REPRESENTATION is a NEW INTERNAL ENCODING of the review                         | REPREZENTACJA to NOWY, WEWNĘTRZNY SPOSÓB ZAPISU recenzji
# created after the input passes through the Dense layer.                           | utworzony po przejściu danych przez warstwę Dense.
#                                                                                   |
# INPUT:                                                                            | WEJŚCIE:
# 10,000 values showing WHICH WORDS appear in the review.                           | 10 000 wartości pokazujących, KTÓRE SŁOWA występują w recenzji.
#                                                                                   |    
# OUTPUT OF THE FIRST DENSE LAYER:                                                  | WYJŚCIE PIERWSZEJ WARSTWY DENSE:
# 16 values forming a 16-DIMENSIONAL INTERNAL REPRESENTATION.                       | 16 wartości tworzących 16-WYMIAROWĄ REPREZENTACJĘ WEWNĘTRZNĄ.
#                                                                                   |
# These 16 values may capture learned features such as:                             | Te 16 wartości może opisywać wyuczone cechy, takie jak:
# POSITIVITY, NEGATIVITY, ACTING QUALITY, EMOTIONAL TONE, etc.                      | POZYTYWNOŚĆ, NEGATYWNOŚĆ, JAKOŚĆ GRY AKTORSKIEJ, EMOCJONALNOŚĆ itd.
#                                                                                   |
# These features are NOT manually defined.                                          | Te cechy NIE SĄ definiowane ręcznie.
# The MODEL LEARNS what each of the 16 dimensions should represent.                 | MODEL SAM UCZY SIĘ, co ma oznaczać każdy z 16 wymiarów.
#
#                                                                                   ↓
#
# The second Dense layer transforms the 16 learned features into 16 NEW features.   | Druga warstwa Dense przekształca 16 wyuczonych cech w 16 NOWYCH cech.
# It combines the previous features using NEW WEIGHTS and applies ReLU.             | Łączy poprzednie cechy za pomocą NOWYCH WAG i stosuje funkcję ReLU.
# This creates a NEW, usually more abstract internal representation.                | Tworzy to NOWĄ, zwykle bardziej abstrakcyjną reprezentację wewnętrzną.
#
#                                                                                   ↓
#
# The final Dense layer transforms 16 features into ONE output value.               | Ostatnia warstwa Dense przekształca 16 cech w JEDNĄ wartość wyjściową.
# The sigmoid activation converts it into a value between 0 and 1.                  | Funkcja sigmoid zamienia ją na wartość od 0 do 1.
# A value close to 0 means a NEGATIVE review.                                       | Wartość bliska 0 oznacza NEGATYWNĄ recenzję.
# A value close to 1 means a POSITIVE review.                                       | Wartość bliska 1 oznacza POZYTYWNĄ recenzję.
model = keras.Sequential(
    [
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Note:
# Crossentropy -> measures the distance between probability distributions or,
# in this case, between the ground-truth distribution and your predictions
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# --- START TRAINING AND VALIDATION DATA
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
# --- END TRAINING AND VALIDATION DATA

# --- START MODEL TRAINING
# Note:
# Validation data is used to evaluate the loss of the model with new data that has not been seen before
# This allows to monitor if model is leaaring new rules or just memorizing the training data
# You can split the data using e.g.
# history = model.fit(
#    x_train,
#    y_train,
#    epochs=20,
#    batch_size=512,
#    validation_split=0.2,
#)
# -> model will use 20% of the training data (x_train and y_train) as validation data 

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
)
# --- END MODEL TRAINING

'''
>>> history_dict = history.history
>>> history_dict.keys()
dict_keys(["accuracy", "loss", "val_accuracy", "val_loss"])
'''

# --- START MODEL EVALUATION
import matplotlib.pyplot as plt

# Plot the training and validation loss
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
# "r--" is for "dashed red line."
plt.plot(epochs, loss_values, "r--", label="Training loss")
# "b" is for "solid blue line."
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("[IMDB] Training and validation loss")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Loss")
plt.legend()
plt.show()

#---
# Clears the figure and plot accuracy
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "r--", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("[IMDB] Training and validation accuracy")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Example predicitons and texts
predictions = model.predict(x_test).flatten()

above_08_indexes = np.where(predictions > 0.8)[0]
below_02_indexes = np.where(predictions < 0.2)[0]

rng = np.random.default_rng(42)

selected_above = rng.choice(above_08_indexes, 10, replace=False)
selected_below = rng.choice(below_02_indexes, 10, replace=False)

def decode_review(sequence):
    return " ".join(
        reverse_word_index.get(index - 3, "?")
        for index in sequence
    )

for index in np.concatenate([selected_above, selected_below]):
    print("Index:", index)
    print("Prediction:", predictions[index])
    print("x_test:", x_test[index])
    print("Decoded text:", decode_review(test_data[index]))
    print("-" * 100)
# --- END MODEL EVALUATION

if __name__ == "__main__":
    pass