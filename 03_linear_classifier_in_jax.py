from jax import numpy as jnp

## --- START LOSS FUNCTION
# In tensorflow we have function
#def training_step(inputs, targets, W, b):
#    # Forward pass, inside of a gradient tape scope
#    # The forward pass records all operations that involve the trainable variables 
#    # (W and b) so that it can compute gradients later
#    with tf.GradientTape() as tape:
#        predictions = model(inputs, W, b)
#        loss = mean_squared_error(predictions, targets)
#    # Retrieves the gradient of the loss with regard to weights
#    # Based on the recorded operations
#    # the gradient tape computes the gradients of the loss with respect to W and b
#    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
#
# --- In JAX we have to implement the function like this
def compute_loss(input_var):
    return jnp.square(input_var)

grad_fn = jax.grad(compute_loss)

input_var = jnp.array(3.0)
grad_of_loss_wrt_input_var = grad_fn(input_var)