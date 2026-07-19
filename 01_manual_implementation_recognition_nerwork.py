# keras.ops is where you will find all the tensor operations you need.
import keras
from keras import ops

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
        x = ops.matmul(inputs, self.W)
        x = x + self.b
        if self.activation is not None:
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
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels

# Train model function
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
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