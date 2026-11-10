import keras

# All Keras layers inherit from the base Layer class
# The difference beetween NaiveDense and SimpleDense:
# NaiveDense is a custom implementation of a dense layer 
#   -> in python rather than using Keras built-in functionality
# SimpleDense is a subclass of keras.Layer
#   -> the base class for all Keras layers
# ---
# Custom layers should inherit from keras.Layer so that Keras can:
# - Automatically track trainable and non-trainable weights
# - Integrate the layer with Model, Sequential, fit(), evaluate(), and predict()
# - Handle automatic building
# - Support saving and loading
# - Propagate training and masking information
# - Manage nested layers and their variables
class SimpleDense(keras.Layer):
    def __init__(self, units, activation=None):
        # Call the base class constructor to set up the layer
        super().__init__()
        # Units is the number of neurons in this layer.
        self.units = units
        # Activation is the function applied to the output of each neuron
        self.activation = activation

    # Weight creation takes place in the build() method.
    def build(self, input_shape):
        # input_shape is a tuple representing the shape of the input tensor
        # For example, for a batch of 128 flattened 28x28 images, input_shape would be (128, 784)
        batch_dim, input_dim = input_shape
        # add_weight is a shortcut method for creating weights. It's
        # also possible to create standalone variables and assign them
        # as layer attributes, like self.W = keras.Variable(shape=...,
        # initializer=...).
        self.W = self.add_weight(
            shape=(input_dim, self.units), initializer="random_normal"
        )
        self.b = self.add_weight(shape=(self.units,), initializer="zeros")

    # We define the forward pass computation in the call() method
    def call(self, inputs):
        y = keras.ops.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

# --- START SOME TESTS
'''
>>> # Instantiates our layer, defined previously
>>> my_dense = SimpleDense(units=32, activation=keras.ops.relu)
>>> # Creates some test inputs
>>> input_tensor = keras.ops.ones(shape=(2, 784))
>>> # Calls the layer on the inputs, just like a function
>>> output_tensor = my_dense(input_tensor)
>>> print(output_tensor.shape)
(2, 32)
↓
# Sample 1 -> 32 output values
# Sample 2 -> 32 output values
#
# Conceptually, the output tensor looks like this:
#
# output_tensor = [
#     [neuron_1_output, ..., neuron_32_output],  # sample 1
#     [neuron_1_output, ..., neuron_32_output],  # sample 2
# ]
#
# Therefore, the output shape is:
# (2, 32)

>>> print("W shape:", my_dense.W.shape)
>>> print("b shape:", my_dense.b.shape)
W shape: (784, 32)
b shape: (32,)
'''
if __name__ == "__main__":
    pass