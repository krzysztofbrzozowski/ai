from collections import defaultdict
import gymnasium as gym
import numpy as np

class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        # ---
        # obs      = (15, 10, False)  # State before action
        # action   = 1                # HIT
        # next_obs = (19, 10, False)  # State after action
        # ---
        # self.q_values[next_obs] = [0.8, 0.2]
        #                           STAND HIT
        # -> q_values 0.8 or 0.2 is EXPECTED SUM OF REWARDS
        #   -> action:  akcja, którą właśnie wykonaliśmy przy 15 punktach.
        #   -> next_obs: to stan, w którym znaleźliśmy się po tej akcji, mając 19 punktów.
        #   -> future_q_value: to oszacowanie wartości najlepszej akcji, którą możemy wykonać z tego nowego stanu.
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        # Rewards: +1 for win, -1 for loss, 0 for draw
        # Let's say we won with STAND and future_q_value
        #   -> future_q_value = not terminated (not 1) = 0
        #   -> target = 1 + 0.95 * 0 = 1
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        # Temporal difference -> error
        # Let's say for action taken (stay), obs = (15, 10, False) we had 0.7 sum of win
        #   -> The difference is 1.0 - 0.7 = 0.3
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        # We are updating the action taken in obs = (15, 10, False)
        #   -> 0.7 + 0.01 * 0.03 = 0.7003
        # Nie przeskoczył od razu do 1.0, ponieważ learning rate kontroluje, jak mocno pojedyncze doświadczenie wpływa na dotychczasową wiedzę.
        # Learning rate is controlling how much one correct experience is influencing current knowledge
        # I właśnie w ten sposób, rozgrywając tysiące gier, agent stopniowo poprawia swoje oszacowania wartości poszczególnych akcji.
        # Thousands of games giving correct estimate
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

        # Here is simpler top overview of learning:
        # # Current estimate: Q(state, action)
        # current_q = self.q_values[obs][action]
        # 
        # # What we actually experienced: reward + discounted future value
        # target = reward + self.discount_factor * max(self.q_values[next_obs])
        # 
        # # How wrong were we?
        # error = target - current_q
        # 
        # # Update estimate: move toward the target
        # new_q = current_q + learning_rate * error

    
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)