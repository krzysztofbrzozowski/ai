from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

import keras
from keras import layers

model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Passing the test digits to be recognized
'''
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
predictions[0]
predictions[0].argmax()
'''
# Show via matplot lib the digit form training data
'''
import matplotlib.pyplot as plt

digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
'''
# Show the label
'''
train_labels[4]
'''


if __name__ == "__main__":
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy:", test_acc)