import keras
# Loads your data, reserving some for validation
from keras.datasets import mnist
from keras import layers

(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

# Creates a model. (We factor this into a separate function so as to
# reuse it later.)
def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model

# --- START CUSTOM TRAINING STEP IN TENSORFLOW
import tensorflow as tf

model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()

def train_step(inputs, targets):
    # Opens a GradientTape
    with tf.GradientTape() as tape:
        # Runs the forward pass
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    # Retrieves the gradients from the tape
    gradients = tape.gradient(loss, model.trainable_weights)
    # Updates the model's trainable weights based on the gradients
    optimizer.apply(gradients, model.trainable_weights)
    return loss

batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]
loss = train_step(inputs, targets)

pass
# --- END CUSTOM TRAINING STEP IN TENSORFLOW
# --- START CUSTOM TRAINING STEP IN PYTORCH
import torch

model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()

def train_step(inputs, targets):
    # Runs the forward pass
    predictions = model(inputs, training=True)
    loss = loss_fn(targets, predictions)
    # Runs the backward pass, populating gradient values
    loss.backward()
    # Recovers the gradient associated with each trainable variable.
    # That weight.value is the PyTorch tensor that contains the
    # variable's value.
    gradients = [weight.value.grad for weight in model.trainable_weights]
    # Updates the model's trainable weights based on the gradients.
    # This must be done in a no_grad() scope.
    with torch.no_grad():
        optimizer.apply(gradients, model.trainable_weights)
    # Don't forget to clear the gradients!
    model.zero_grad()
    return loss

batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]
loss = train_step(inputs, targets)

pass
# --- END CUSTOM TRAINING STEP IN PYTORCH
# --- START CUSTOM TRAINING STEP IN JAX
import jax

model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# Gradients are computed for the entries in the first argument
# (trainable_variables here)
def compute_loss_and_updates(
    trainable_variables,
    non_trainable_variables,
    inputs,
    targets
):
    # Calls stateless_call
    outputs, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, inputs, training=True
    )
    loss = loss_fn(targets, outputs)
    # Returns the scalar loss value and the updated non-trainable
    # weights
    return loss, non_trainable_variables

optimizer = keras.optimizers.Adam()
optimizer.build(model.trainable_variables)

# Pass to to jax.value_and_grad() the function that computes the loss and
# the updated non-trainable weights. The has_aux=True argument indicates
# that the function returns a tuple, where the first element is the scalar
# loss value and the second element is the updated non-trainable weights
grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)

def train_step(state, inputs, targets):
    # Unpacks the state
    (trainable_variables, non_trainable_variables, optimizer_variables) = state
    # Computes gradients and updates to non-trainable variables
    (loss, non_trainable_weights), gradients = grad_fn(
        trainable_variables, non_trainable_variables, inputs, targets
    )
    # Updates trainable variables and optimizer variables
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    return loss, (
        # Returns the updated state alongside the loss
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )

batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]

trainable_variables = [v.value for v in model.trainable_variables]
non_trainable_variables = [v.value for v in model.non_trainable_variables]
optimizer_variables = [v.value for v in optimizer.variables]

state = (trainable_variables, non_trainable_variables, optimizer_variables)
loss, state = train_step(state, inputs, targets)
# --- END CUSTOM TRAINING STEP IN JAX
