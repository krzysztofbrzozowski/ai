import numpy as np
import tensorflow as tf

# --- START GENERATE OF SAMPLE DATA
# 2 types of data, lineary separable
num_samples_per_class = 1000
# Generating some data using np
negative_samples = np.random.multivariate_normal(
    # Generates the first class of points: 1,000 random 2D points with
    # specified "mean" and "covariance matrix." Intuitively, the
    # "covariance matrix" describes the shape of the point cloud, and
    # the "mean" describes its position in the plane. `cov=[[1,
    # 0.5],[0.5, 1]]` corresponds to "an oval-like point cloud oriented
    # from bottom left to top right."
    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)
positive_samples = np.random.multivariate_normal(
    # Generates the other class of points with a different mean and the
    # same covariance matrix (point cloud with a different position and
    # the same shape)
    mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)

# Stack samples in 2D array (2000, 2)
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
# Create corresponding labes 1D array (2000, 1)
targets = np.vstack(
    (
        np.zeros((num_samples_per_class, 1), dtype="float32"),
        np.ones((num_samples_per_class, 1), dtype="float32"),
    )
)
# --- END GENERATE OF SAMPLE DATA
# --- START PLOT DATA
import matplotlib.pyplot as plt

# c -> color the input class based on the target class
plt.scatter(
    inputs[:, 0],
    inputs[:, 1],
    c=targets[:, 0]
)
plt.show()
# --- END PLOT DATA

# --- START LINEAR CLASSIFIER
# The inputs will be 2D points.
input_dim = 2
# The output predictions will be a single score per sample (close to 0
# if the sample is predicted to be in class 0, and close to 1 if the
# sample is predicted to be in class 1).
output_dim = 1
# Since we will do the matmul we can create shape of W(1,2)
# W = [[10, 11]]
# do predictions = tf.matmul(inputs, tf.transpose(W)) + b
#
# But probably better is to use
# Already predefined version of W
# 
# Rule of thumb:
# -> internal diamentions needs to be equal
# (2000, 2) @ (2, 1) = (2000, 1)
#        ↑     ↑
#        └──2──┘
# -> external diamentions gives the output shape

W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))
# --- END LINEAR CLASSIFIER

# --- START OF THE MODEL
def model(inputs, W, b):
    return tf.matmul(inputs, W) + b
# --- END OF THE MODEL

# --- START LEARNING
def mean_squared_error(targets, predictions):
    # First, calculate the difference between the expected target and
    # the model's prediction separately for every sample.
    #
    # Example:
    #     targets:     [[0.0], [1.0], [1.0], [0.0]]
    #     predictions: [[0.2], [0.8], [0.4], [0.1]]
    #
    # The differences are:
    #     0.0 - 0.2 = -0.2
    #     1.0 - 0.8 =  0.2
    #     1.0 - 0.4 =  0.6
    #     0.0 - 0.1 = -0.1
    #
    # After calculating the differences, `tf.square()` squares each
    # value independently:
    #     (-0.2)^2 = 0.04
    #     ( 0.2)^2 = 0.04
    #     ( 0.6)^2 = 0.36
    #     (-0.1)^2 = 0.01
    #
    # Squaring makes every error non-negative and penalizes larger
    # prediction errors more strongly. The resulting tensor has the
    # same shape as `targets` and `predictions`:
    #
    #     [[0.04],
    #      [0.04],
    #      [0.36],
    #      [0.01]]
    #
    # Shape example: (4, 1) -> (4, 1).
    per_sample_losses = tf.square(targets - predictions)

    # Calculate the mean of all squared errors and reduce the entire
    # `per_sample_losses` tensor to a single scalar loss value.
    #
    # For the squared errors above, `tf.reduce_mean()` computes:
    #
    #     (0.04 + 0.04 + 0.36 + 0.01) / 4 = 0.1125
    #
    # For a tensor with shape (2000, 1), this operation averages the
    # squared errors across all 2,000 samples and returns a scalar tensor
    # with shape (). This single value represents the model's average
    # prediction error for the entire dataset or current batch.
    #
    # During training, this scalar loss is differentiated with respect
    # to W and b. The resulting gradients indicate how W and b should
    # be changed to reduce the average prediction error.
    return tf.reduce_mean(per_sample_losses)

learning_rate = 0.1

# Wraps the function in a tf.function decorator to speed it up
@tf.function(jit_compile=True)
def training_step(inputs, targets, W, b):
    # Forward pass, inside of a gradient tape scope
    # The forward pass records all operations that involve the trainable variables 
    # (W and b) so that it can compute gradients later
    with tf.GradientTape() as tape:
        predictions = model(inputs, W, b)
        loss = mean_squared_error(predictions, targets)
    # Retrieves the gradient of the loss with regard to weights
    # Based on the recorded operations
    # the gradient tape computes the gradients of the loss with respect to W and b
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    # Updates the weights
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss

# Do the actual learning
for step in range(40):
    loss = training_step(inputs, targets, W, b)
    print(f"Loss at step {step}: {loss:.4f}")
# --- END LEARNING

# --- START VERIFICATION
# Here is starting simple verification of trained W and b
predictions = model(inputs, W, b)
# The target classes are encoded as 0 and 1:
#
#     class 0 -------- 0.5 -------- class 1
#
# Since 0.5 is exactly halfway between 0 and 1, it is used as the
# decision threshold. Predictions less than or equal to 0.5 are assigned
# to class 0, while predictions greater than 0.5 are assigned to class 1.
# 
# class 1: prediction > 0.5
# 
#         •   •   •
#       •   •   •
# ---------------------- decision boundary: prediction = 0.5
#   •   •   •
#     •   •   •
# 
# class 0: prediction < 0.5
# 
predicted_classes = predictions[:, 0] > 0.5
# W can change the equation from:
# 1. w1​x + w2​y + b = 0.5
# to
# 2. y = ax + c
# Note the y in 1. is the same y in 2.
#
# After the transformation we will have 
# y = −(w2/​w1​​)x+​(0.5−b)/w2​​

# Generates 100 regularly spaced numbers between -1 and 4, which we
# will use to plot our line -> x values to draw the line
x = np.linspace(-1, 4, 100)
# This is our line's equation
# scalar -> (-W[0] / W[1]) * array -> x + scalar -> (0.5 - b) / W[1]
# y <- this is array 100 (numpy boradcasting)
y = -W[0] / W[1] * x + (0.5 - b) / W[1]

plt.plot(x, y, "-r")
# Plots our model's predictions on the same plot
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
# --- END VERIFICATION

if __name__ == '__main__':
    pass