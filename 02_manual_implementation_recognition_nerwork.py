# keras.ops is where you will find all the tensor operations you need.
import keras
from keras import ops
import math

# --- Start test data for the model
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
# --- End test data for the model

# We are creating 2 layers of the network
# one with 512 neurons -> ReLU activation
# other with 10 neurons -> Softmax activation
class NaiveDense:
    '''
    A naive implementation of a dense layer
    example call: NaiveDense(input_size=28 * 28, output_size=512, activation=ops.relu)
    '''
    def __init__(self, input_size, output_size, activation=None):
        self.activation = activation
        self.W = keras.Variable(
            # Creates a matrix W of shape (input_size, output_size),
            # initialized with random values drawn from a uniform
            # distribution
            shape=(input_size, output_size), initializer="uniform"
        )
        # Creates a vector b of shape (output_size,), initialized with
        # zeros
        self.b = keras.Variable(shape=(output_size,), initializer="zeros")

    # Applies the forward pass
    def __call__(self, inputs):
        # Multiplies the entire batch by the same shared weight matrix.
        # Each row in `inputs` represents one flattened image, and each column
        # in `self.W` contains the weights of one neuron.
        # Example: (128, 784) @ (784, 512) -> (128, 512),
        # producing 512 neuron outputs for each of the 128 images.
        x = ops.matmul(inputs, self.W)
        x = x + self.b
        if self.activation is not None:
            # Applies the activation function element-wise to the entire output tensor.
            # For ReLU, each value is replaced with max(value, 0).
            # Example: the shape remains unchanged: (128, 512) -> (128, 512).
            x = self.activation(x)
        return x

    @property
    # The convenience method for retrieving the layer's weights
    def weights(self):
        return [self.W, self.b]


class NaiveSequential:
    '''
    A naive implementation of a sequential model
    '''
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


class BatchGenerator:
    '''
    A generator for creating batches of images and labels
    '''
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        # Images are the ndim 2 -> so the output will be 128x784
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels

# --- Start actual learning/update part
# Below is manual implementation of the update gradient function, regarding book:
# In practice, you will almost never implement a weight update step like this by hand
# Instead, you would use an Optimizer instance from Keras — like this -> go to
# NEW KERAS OPTIMIZER
# 
# --- START OLD OPTIMIZER
# learning_rate = 1e-3
# def update_weights(gradients, weights):
#     for g, w in zip(gradients, weights):
#         # Assigns a new value to the variable, in place
#         w.assign(w - g * learning_rate)
# --- END OLD OPTIMIZER

# --- START NEW KERAS OPTIMIZER
from keras import optimizers

optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))
# --- END NEW KERAS OPTIMIZER


# This is most important function -> based on the labes
# The weights and biases are corrected
# Basically it is called in for loop depending on train images / batch size -> number of iterations
# for batch_counter in range(batch_generator.num_batches):
#       loss = one_training_step(model, images_batch, labels_batch
# --- START OLD IMPLEMENTATION one_training_step 
# def one_training_step(model, images_batch, labels_batch):
#     # Runs the "forward pass"
#     # Here the all magic happenig -> we are producing current predictions per one batch
#     # See NaiveDense -> 
#     #   __call__ (called on model object call):
#     #       x = ops.matmul(inputs, self.W)
#     #       x = x + self.b
#     predictions = model(images_batch)
#     # Calculate the loss and average loss
#     loss = ops.sparse_categorical_crossentropy(labels_batch, predictions)
#     average_loss = ops.mean(loss)
#     # Computes the gradient of the loss with regard to the weights. The
#     # output, gradients, is a list where each entry corresponds to a
#     # weight from the model.weights list. We haven't defined this
#     # function yet!
#     gradients = get_gradients_of_loss_wrt_weights(loss, model.weights)
#     # Updates the weights using the gradients. We haven't defined this
#     # function yet!
#     update_weights(gradients, model.weights)
#     return loss
# --- END OLD IMPLEMENTATION one_training_step

# --- START NEW IMPLEMENTATION one_training_step
import tensorflow as tf

def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        loss = ops.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = ops.mean(loss)
    # 'tape' gradient calculation for backpropagation
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss
# --- END NEW IMPLEMENTATION one_training_step

# Train model function
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            # Here is actual training happening
            # Model takes the batch_size training images and according labels
            # The update of weights and the biases is updated after each bach iteration (e.g. 128 steps)
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")

if __name__ == "__main__":
    model = NaiveSequential(
        [
            NaiveDense(input_size=28 * 28, output_size=512, activation=ops.relu),
            NaiveDense(input_size=512, output_size=10, activation=ops.softmax),
        ]
    )
    # Verify self.W and self.b are created for each layer (2 layers * 2 weights = 4)
    assert len(model.weights) == 4

    # Train the model
    fit(model, train_images, train_labels, epochs=10, batch_size=128)
    pass
    
    # Verification steps 
    '''
    >>> predictions = model(test_images)
    >>> predicted_labels = ops.argmax(predictions, axis=1)
    >>> matches = predicted_labels == test_labels
    >>> f"accuracy: {ops.mean(matches):.2f}"
    '''