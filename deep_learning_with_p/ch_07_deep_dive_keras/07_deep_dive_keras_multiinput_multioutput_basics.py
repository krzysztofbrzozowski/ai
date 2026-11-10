import keras
from keras import layers

# --- START OF SEQUENTIONAL APPROACH (end line 112)
model = keras.Sequential()
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

pass

# --- SOME BASIC KNOWLEDGE
"""
>>> # At that point, the model isn't built yet.
>>> model.weights
[]
"""
# Note:
# The input contains 3 features, and the Dense layer contains 2 neurons.
# Every input feature is connected to every neuron, so the weight matrix
# has shape (3, 2): 3 input features × 2 neurons.
#
# Example: 3 input features -> 2 neurons
#
#   Neuron 1   Neuron 2
#      ↓          ↓
# weights = [
#   [ 0.5,       0.1],   # ← Input feature 1
#   [ 1.0,      -0.5],   # ← Input feature 2
#   [ -0.5,      0.25],  # ← Input feature 3
# ]
#
# Each row corresponds to one input feature.
# Therefore, the number of rows is equal to the number of input features.
#
# Each column corresponds to one neuron.
# Therefore, each neuron has its own column containing one weight
# for every input feature.
#
# The layer computes:
#
# output = input @ weights + bias
#
# For one input sample with shape (3,), the output has shape (2,),
# because each of the two neurons produces one value.

# --- BUILD MODEL (NOT SEQUENTIAL YET)
model.build(input_shape=(None, 3))
"""
>>> model.build(input_shape=(None, 3))
>>> # Now you can retrieve the model's weights.
>>> model.weights
[<Variable shape=(3, 64), dtype=float32, path=sequential/dense_2/kernel ...>,
 <Variable shape=(64,), dtype=float32, path=sequential/dense_2/bias ...>,
 <Variable shape=(64, 10), dtype=float32, path=sequential/dense_3/kernel ...>,
 <Variable shape=(10,), dtype=float32, path=sequential/dense_3/bias ...>>]
"""
pass

# --- SOME INFO ABOUT MODEL
model.summary()
#Model: "sequential"
#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
#┃ Layer (type)                        ┃ Output Shape               ┃        Param # ┃
#┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
#│ dense (Dense)                       │ (None, 64)                 │            256 │
#├─────────────────────────────────────┼────────────────────────────┼────────────────┤
#│ dense_1 (Dense)                     │ (None, 10)                 │            650 │
#└─────────────────────────────────────┴────────────────────────────┴────────────────┘
# Total params: 906 (3.54 KB)
# Trainable params: 906 (3.54 KB)
# Non-trainable params: 0 (0.00 B)
# The first Dense layer receives 3 input features and contains 64 neurons.
# Additonal note:
# The None in the tensor shapes represents the batch size: this model allows batches of any size
# Why?
# Each of the 64 neurons has:
# - 3 weights: one for each input feature
# - 1 bias
#
# First layer parameter count:
#     weights: 3 * 64 = 192
#     biases:          64
#     total:  192 + 64 = 256
#
# The second Dense layer receives 64 values from the previous layer
# and contains 10 neurons.
#
# Each of the 10 neurons has:
# - 64 weights: one for each input from the previous layer
# - 1 bias
#
# Second layer parameter count:
#     weights: 64 * 10 = 640
#     biases:             10
#     total:     640 + 10 = 650
#
# Total number of trainable parameters:
#     256 + 650 = 906
#
# ReLU and softmax do not add any trainable parameters.

# --- IT IS POSSIBLE TO BUILD MODEL SEQUENTIALY -> HELPFUL WITH DEBUG
model = keras.Sequential()
# Use an Input to declare the shape of the inputs. Note that the shape
# argument must be the shape of each sample, not the shape of one
# batch.
model.add(keras.Input(shape=(3,)))
model.add(layers.Dense(64, activation="relu"))
model.summary()
model.add(layers.Dense(10, activation="softmax"))
model.summary()
# --- END OF SEQUENTIONAL APPROACH
# --- BEGINING OF MULTI-INPUT AND MULTI-OUTPUT MODEL
# Simple example what has been done before
inputs = keras.Input(shape=(3,), name="my_input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs, name="my_functional_model")

# System to rank customer support tickets by priority and route them to the appropriate department
# INPUTS (3):
# -> The title of the ticket (text input)
# -> The text body of the ticket (text input)
# -> Any tags added by the user (categorical input, assumed here to be multi-hot encoded)
# OUTPUTS (2):
# -> The priority score of the ticket, a scalar between 0 and 1 (sigmoid output)
# -> The department that should handle the ticket (a softmax over the set of departments)
vocabulary_size = 10000
num_tags = 100
num_departments = 4

# Defines model inputs (3)
title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

# Combines input features into a single tensor, features, by
# concatenating them
features = layers.Concatenate()([title, text_body, tags])

# Applies intermediate layer to recombine input features into richer
# representations
features = layers.Dense(64, activation="relu", name="dense_features")(features)

# Defines model outputs (2)
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(
    num_departments, activation="softmax", name="department"
)(features)

# Creates the model by specifying its inputs and outputs
model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department],
)

# --- START MODEL TRAINING
import numpy as np

num_samples = 1280

# Dummy input data
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# Dummy target data
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, num_departments, size=(num_samples, 1))

# Version with the parameters provided only by order, so the loss and metrics are assigned to the outputs in the order they were passed to the model
# model.compile(
#     optimizer="adam",
#     loss=["mean_squared_error", "sparse_categorical_crossentropy"],
#     metrics=[["mean_absolute_error"], ["accuracy"]],
# )
# model.fit(
#     [title_data, text_body_data, tags_data],
#     [priority_data, department_data],
#     epochs=1,
# )
# model.evaluate(
#     [title_data, text_body_data, tags_data], [priority_data, department_data]
# )
# priority_preds, department_preds = model.predict(
#     [title_data, text_body_data, tags_data]
# )

# or with stricly assigned output parameters -> loss + metrics are assigned to the output names
model.compile(
    optimizer="adam",
    loss={
        "priority": "mean_squared_error",
        "department": "sparse_categorical_crossentropy",
    },
    metrics={
        "priority": ["mean_absolute_error"],
        "department": ["accuracy"],
    },
)
model.fit(
    {
        "title": title_data,
        "text_body": text_body_data,
        "tags": tags_data
    },
    {
        "priority": priority_data,
        "department": department_data
    },
    epochs=1,
)
model.evaluate(
    {
        "title": title_data,
        "text_body": text_body_data,
        "tags": tags_data
    },
    {
        "priority": priority_data,
        "department": department_data
    },
    # To get dict as return this needs to be set
    return_dict=True,
)
priority_preds, department_preds = model.predict(
    {
        "title": title_data,
        "text_body": text_body_data,
        "tags": tags_data
    }
)
# --- END OF MULTI-INPUT AND MULTI-OUTPUT MODEL
# --- START MODEL GRAPH
keras.utils.plot_model(model, "ticket_classifier.png")

# More detailed view
keras.utils.plot_model(
    model,
    "ticket_classifier_with_shape_info.png",
    show_shapes=True,
    show_layer_names=True,
)

if __name__ == "__main__":
    pass
