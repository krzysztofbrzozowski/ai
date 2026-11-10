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

# --- We can during computation get also the loss value and the gradient at the same time
grad_fn = jax.value_and_grad(compute_loss)
output, grad_of_loss_wrt_input_var = grad_fn(input_var)

# --- Computing the loss with regard to a, b, b
# state contains a, b, and c. It must be the first argument.
def compute_loss(state, x, y):
    ...
    return loss

grad_fn = jax.value_and_grad(compute_loss)
state = (a, b, c)
# grads_of_loss_wrt_state has the same structure as state.
loss, grads_of_loss_wrt_state = grad_fn(state, x, y)
# --- Adding additional output of the metafunction wirh has_aux=True
def compute_loss(state, x, y):
    ...
    # Returns a tuple
    return loss, output

# Passes has_aux=True here
grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
# Gets back a nested tuple
loss, (grads_of_loss_wrt_state, output) = grad_fn(state, x, y)