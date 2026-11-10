from keras.datasets import reuters
import keras

# --- START TRAINING DATA
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000
)

# We can decode data back to text using the word index
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = " ".join(
    # The indices are offset by 3 because 0, 1, and 2 are reserved
    # indices for "padding," "start of sequence," and "unknown."
    [reverse_word_index.get(i - 3, "?") for i in train_data[10]]
)
import numpy as np

# Vectorize input data
def multi_hot_encode(sequences, num_classes):
    """
    - Example input:
    sequences = [
        [2, 5],
        [1, 2, 7]
    ]
    - Initial results matrix:
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    - After multi-hot encoding:
    [
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1]
    ]
    Creates an all-zero matrix of shape (len(sequences), num_classes)
    """
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

def one_hot_encode(labels, num_classes=46):
    """
    Put 1s in the approprate place
    e.g. if the label is 3, then the result will be [0, 0, 0, 1, 0, 0, ...]
    """
    results = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        results[i, label] = 1.0
    return results

# Vectorized training labels
y_train = one_hot_encode(train_labels)
# Vectorized test labels
y_test = one_hot_encode(test_labels)

# It is possible to use the built-in Keras function to one-hot encode the labels
# It is not as efficient as the above implementation
from keras.utils import to_categorical

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
# --- END TRAINING DATA
# --- START MODEL DEFINITION
import keras
from keras import layers

model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax"),
    ]
)
# --- END MODEL DEFINITION
# --- START MODEL COMPILATION
top_3_accuracy = keras.metrics.TopKCategoricalAccuracy(
    k=3, name="top_3_accuracy"
)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy", top_3_accuracy],
)
# --- END MODEL COMPILATION
# --- START MODEL TRAINING
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
)
# --- END MODEL TRAINING
# --- START MODEL EVALUATION
import matplotlib.pyplot as plt
# --- loss
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Loss")
plt.legend()
plt.show()
# --- accuracy
plt.clf()
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "r--", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# --- top-3 accuracy
plt.clf()
acc = history.history["top_3_accuracy"]
val_acc = history.history["val_top_3_accuracy"]
plt.plot(epochs, acc, "r--", label="Training top-3 accuracy")
plt.plot(epochs, val_acc, "b", label="Validation top-3 accuracy")
plt.title("Training and validation top-3 accuracy")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Top-3 accuracy")
plt.legend()
plt.show()
# --- END MODEL EVALUATION

# --- START MODEL EVALUATION ON TEST DATA
model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    x_train,
    y_train,
    epochs=9,
    batch_size=512,
)
results = model.evaluate(x_test, y_test)

"""
>>> import copy
>>> test_labels_copy = copy.copy(test_labels)
>>> np.random.shuffle(test_labels_copy)
>>> hits_array = np.array(test_labels == test_labels_copy)
>>> hits_array.mean()
0.18655387355298308
-> Side info: if we will shuffle data and compare it with the original data, we will get a random accuracy of ~0.18
-> So the model accuracy of ~0.78 is much better than random guessing
Note:
RANDOM BASELINE ACCURACY
-> it is a base model we can use to compare our model with, to see if it is better than random guessing
"""
# --- END MODEL EVALUATION ON TEST DATA
# --- START MODEL PREDICTION ON TEST DATA
predictions = model.predict(x_test)
"""
>>> predictions[0].shape
(46,)
>>> np.sum(predictions[0])
1.0
>>> np.argmax(predictions[0])
4
"""
# --- END MODEL PREDICTION ON TEST DATA
# --- START MODEL EVALUATION WITH SPARSE CATEGORICAL CROSSENTROPY
# -> A different way to handle the labels and the loss
# y_train = train_labels
# y_test = test_labels

# model = keras.Sequential(
#     [
#         layers.Dense(64, activation="relu"),
#         layers.Dense(64, activation="relu"),
#         layers.Dense(46, activation="softmax"),
#     ]
# )
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"],
# )
# model.fit(
#     x_train,
#     y_train,
#     epochs=9,
#     batch_size=512,
# )
# results = model.evaluate(x_test, y_test)
# # [0.9212439656257629, 0.800979495048523]
# --- END MODEL EVALUATION WITH SPARSE CATEGORICAL CROSSENTROPY



if __name__ == "__main__":
    pass