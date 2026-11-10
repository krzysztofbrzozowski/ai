import jax
from jax import numpy as jnp
import numpy as np

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


# --- START OF THE MODEL
def model(inputs, W, b):
    return jnp.matmul(inputs, W) + b
# --- END OF THE MODEL

# --- START LEARNING
def mean_squared_error(targets, predictions):
    per_sample_losses = jnp.square(targets - predictions)
    return jnp.mean(per_sample_losses)

# state -> tensors we need to calculae the loss from
def compute_loss(state, inputs, targets):
    W, b = state
    predictions = model(inputs, W, b)
    loss = mean_squared_error(targets, predictions)
    return loss

grad_fn = jax.value_and_grad(compute_loss)
learning_rate = 0.1
# We use the jax.jit decorator to take advantage of XLA compilation.
@jax.jit
def training_step(inputs, targets, W, b):
    # Computes the forward pass and backward pass in one go
    loss, grads = grad_fn((W, b), inputs, targets)
    grad_wrt_W, grad_wrt_b = grads
    # Updates W and b
    W = W - grad_wrt_W * learning_rate
    b = b - grad_wrt_b * learning_rate
    # Make sure to return the new values of W and b in addition to the
    # loss!
    # Needed because the outside valueas are immutable
    # thats why we need to assing them again
    return loss, W, b


input_dim = 2
output_dim = 1

W = jax.numpy.array(np.random.uniform(size=(input_dim, output_dim)))
b = jax.numpy.array(np.zeros(shape=(output_dim,)))

state = (W, b)
for step in range(40):
    loss, W, b = training_step(inputs, targets, W, b)
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

# In generall we need to classify the data 0 or 1
# We have the inputs(X) and targets(Y_true -> those are true because those are training data)
# 1. We need to find such W and b's to make linear W @ X = Y_true
# This is training_step:
#   ...
#   -> This exec stars recording the all comutations
#       required to further autodiffrentiation
#   loss, grads = grad_fn((W, b), inputs **X**, targets **Y_true**)
#   ↓
#
#
# 2. We need gradients to update W and b
#    -> reminder: dLoss/dW (or dLoss/db) = gradient
#    -> Here value_and_grad performs automatic differentiation
#    -> Since W and b are contained in the first argument (state),
#       JAX can differentiate the loss with respect to both W and b.
#    -> JAX traces all operations that transform W and b into the
#       final loss value and then applies reverse-mode autodiff
#       (conceptually similar to a tape).
#
#   grad_fn = jax.value_and_grad(compute_loss)  
#   ↓
#
#   def compute_loss(state, inputs, targets):
#       W, b = state
#       predictions = model(inputs, W, b) <- **Y_predicitons**
#       loss = mean_squared_error(targets **Y_true**, predictions **Y_predictions**)
#       return loss
#
#   ↓
#   3. Every pass W, b are updated
#   W = W - grad_wrt_W * learning_rate
#   b = b - grad_wrt_b * learning_rate
#   ...
#   