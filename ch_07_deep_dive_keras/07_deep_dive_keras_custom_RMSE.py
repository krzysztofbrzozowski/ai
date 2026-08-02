import keras

from keras.datasets import mnist
from keras import layers

# Creates a model. (We factor this into a separate function so as to
# reuse it later.)
def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model

# Loads your data, reserving some for validation
(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

# --- CALL THIS BEFORE CUSTOM RMSE IMPLEMENTATION
# model = get_mnist_model()
# # Compiles the model by specifying its optimizer, the loss function to
# # minimize, and metrics to monitor
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"],
# )
# # Uses `fit()` to train the model, optionally providing validation data
# # to monitor performance on unseen data
# model.fit(
#     train_images,
#     train_labels,
#     epochs=3,
#     validation_data=(val_images, val_labels),
# )
# # Uses `evaluate()` to compute the loss and metrics on new data
# test_metrics = model.evaluate(test_images, test_labels)
# # Uses `predict()` to compute classification probabilities on new data
# predictions = model.predict(test_images)
# --- 

# --- START CUSTOM METRICS (MSE) IMPLEMENTATION
from keras import ops

# Subclasses the Metric class
class RootMeanSquaredError(keras.metrics.Metric):
    # Defines the state variables in the constructor. Like for layers,
    # you have access to the add_weight() method.
    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
        self.total_samples = self.add_weight(
            name="total_samples", initializer="zeros"
        )

    # Implements the state update logic in update_state(). The y_true
    # argument is the targets (or labels) for one batch, while y_pred
    # represents the corresponding predictions from the model. To match
    # our MNIST model, we expect categorical predictions and integer
    # labels. You can ignore the sample_weight argument; we won't use
    # it here.
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.one_hot(y_true, num_classes=ops.shape(y_pred)[1])
        mse = ops.sum(ops.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = ops.shape(y_pred)[0]
    
    # Rerturn the RMSE
    def result(self):
        return ops.sqrt(self.mse_sum / self.total_samples)

    # Reset the state between batches
    def reset_state(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0.)

# --- CALL THIS AFTER CUSTOM RMSE IMPLEMENTATION
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", RootMeanSquaredError()],
)
model.fit(
    train_images,
    train_labels,
    epochs=3,
    validation_data=(val_images, val_labels),
)
test_metrics = model.evaluate(test_images, test_labels)
# ---