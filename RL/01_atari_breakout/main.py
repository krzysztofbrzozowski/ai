# --- START SETUP
import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

import numpy as np
import tensorflow as tf

# We are importing model and model_target
from model import *

# --- START GLOBAL VARIABLES
# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000
max_episodes = 10000  # Limit training episodes, will run until solved if smaller than 1
model_weights_path = Path(__file__).resolve().parent / "breakout.weights.h5"
# ---
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# --- END GLOBAL VARIABLES

# Continue learning from saved weights when available.
if model_weights_path.is_file():
    model.load_weights(model_weights_path)
    print("Loaded model weights from {}".format(model_weights_path))
else:
    print("No saved model weights found; starting training from scratch.")

# Start the target network with the same weights as the main model.
model_target.set_weights(model.get_weights())

# Register Atari environments
gym.register_envs(ale_py)

# Create the Breakout environment without built-in frame skipping
# Set render_mode="human" to display the game in a window
env = gym.make(
    "ALE/Breakout-v5",
    frameskip=1,
    repeat_action_probability=0.0,
    render_mode="human",
)

# Environment preprocessing:
# -> Each action is repeated for 4 consecutive game frames
# -> Images are converted to grayscale
# -> Images are resized to 84x84 pixels
env = AtariPreprocessing(env, frame_skip=4)

# Stack the last 4 observations to capture motion information
env = FrameStackObservation(env, stack_size=4)

# The two wrappers above produce the following observations:
#
# Game frames:   1   2   3   4   5   6   7   8   9  10  11  12
#                └───────────┘   └───────────┘   └───────────┘
# Action:              A               B               C
# Observation:               ↑               ↑               ↑
#
# Each observation contains the last 4 preprocessed images.
# Observation shape: (4, 84, 84)
#
# Frame skip = 4: The agent selects a new action every 4 game frames.
# Frame stack = 4: The agent observes the last 4 processed images.

# Initialize the environment with a fixed random seed
env.action_space.seed(seed)

print("Observation shape:", env.observation_space.shape)  # (4, 84, 84)
# --- END SETUP

# Use epsilon-greedy for exploration
# In DQN case, instead of taking value from Q-table, CNN giving us the prediction
def get_action(state):
    print(f"epsilon: {epsilon}, frame_count: {frame_count}")
    if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
        # Take random action
        print(f"random action")
        return np.random.choice(num_actions)
    else:
        print("model prediction")
        # Predict action Q-values
        # From environment state
        # Convert observations to the tensor
        state_tensor = keras.ops.convert_to_tensor(state)
        # Expand the state tensor with one new diamention
        # -> state_tensor.shape  # (1, 4, 84, 84)
        state_tensor = keras.ops.expand_dims(state_tensor, 0)
        # It is similar to action_probs = model.predict(state_tensor)
        # model.predict() introduces additional overhead:
        # -> Validates and prepares input data
        # -> Creates and manages batches
        # -> Handles the prediction loop
        # -> Converts output tensors to NumPy arrays
        #
        # For a single observation, calling model() directly avoids
        # much of this overhead and is usually faster.
        # ---
        # action_probs = [[1.2, 0.5, 4.8, 2.1]]
        action_probs = model(state_tensor, training=False)
        # Take best action (NOOP, FIRE, RIGHT or LEFT)
        return keras.ops.argmax(action_probs[0]).numpy()

# --- START MAIN LOOP/TRAINING
# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
best_running_reward = -float("inf")
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

while True:
    obs, info = env.reset(seed=seed)

    state = np.array(obs)
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        # Get the best possible action if epsilon is small (exploitation)
        action = get_action(state=state)

        # Decay probability of taking random action
        # After some random moves
        # TODO It might be needed to use this as global or maybe add this to calass like in RL 00
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the action in our environment, it is similar
        # to obs_next from previous example
        # ---
        # Till here we have few mine steps
        # 1. Observation of the game
        #   -> at fisrt the observation is taken from reset env.reset(seed=seed)
        #   -> later on the observation is taken from state = state_next 
        # 2. Get action (CNN model)
        # 3. Execute the step from prediction in env.step(action)
        # 4. Go to point 1 (state = state_next)
        state_next, reward, terminated, _, _ = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        # Why do we use a Replay Buffer -> this is collection of training samples?
        #   -> We are storing each move of the game and history
        #   -> Later on we are randomizing the move choice np.random.choice
        #       -> thats why the corelation between conse
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(terminated)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        # The model is trained on batches of 32 experiences every 4 agent steps.
        # Since AtariPreprocessing uses frame_skip=4, the model is updated
        # approximately every 16 game frames.
        if frame_count % update_after_actions == 0 \
        and len(done_history) > batch_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = keras.ops.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            # Model is trying to predict few samples
            # TODO, why? I don't understand this
            # model_target is copy of main model -> model
            # state_next_sample.shape() -> (32, 4, 84, 84) -> we have 32 observations
            # future_rewards.shape() -> (32, 4) -> we have 4 Q-Values for 32 observations
            # future_rewards = [
            #     [1.0, 2.0, 3.5, 0.5],  # Experience 1: NOOP, FIRE, RIGHT, LEFT
            #     [2.0, 1.0, 0.5, 1.5],  # Experience 2
            #     [0.5, 4.2, 1.0, 2.0],  # Experience 3
            # ]
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            # keras.ops.amax(future_rewards, axis=1) -> max value from the future reward
            # Select the highest Q-value for each experience
            # max_future_q = keras.ops.amax(future_rewards, axis=1)
            # max_future_q.shape     (32,)
            # updated_q_values.shape (32,)
            updated_q_values = rewards_sample + gamma * keras.ops.amax(
                future_rewards, axis=1
            )
            
            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample
            
            # Create a mask so we only calculate loss on the updated Q-values
            masks = keras.ops.one_hot(action_sample, num_actions)

            # Calculate loss function 
            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = keras.ops.sum(keras.ops.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                # 32 current q_action values how different with next_state updated 32 q_action values
                # We have loss
                # updated_q_values  → y_true, the target values calculated using model_target.
                # q_action          → y_pred, the Q-values predicted by our model.
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            # Backpropagete the gradients to the model
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Update model target each ~10k samples
        # We are updating every 10k to avoid situation model is constantly changing the target
        # In supervised learning
        #   -> loss = loss_function(y_true, y_pred)
        # In DQN
        #   -> loss = loss_function(updated_q_values, q_action)
        # So we need to have more or less constant target to train the model
        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        # Limit the state and reward history
        # Remove the oldest element from history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]
        
        
        # Break for loop
        if terminated:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    # Keep the weights with the highest average reward during this training run.
    if running_reward > best_running_reward:
        model.save_weights(model_weights_path)
        best_running_reward = running_reward
        print(
            "Saved best model: average reward {:.2f} at episode {} to {}".format(
                best_running_reward, episode_count, model_weights_path
            )
        )

    if running_reward > 40:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

    if (
        max_episodes > 0 and episode_count >= max_episodes
    ):  # Maximum number of episodes reached
        print("Stopped at episode {}!".format(episode_count))
        break
