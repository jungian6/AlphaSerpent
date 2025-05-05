import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import os


class DQNAgent:
    def __init__(self, state_shape, action_size):
        """Initialize the DQN Agent with parameters and neural network models"""
        # Environment parameters
        self.state_shape = state_shape  # Shape of the game state (grid height, grid width, channels)
        self.action_size = action_size  # Number of possible actions (4 for snake: up, right, down, left)

        # Learning parameters
        self.gamma = 0.95  # Discount factor for future rewards
        self.learning_rate = 0.001  # Learning rate for the neural network
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Rate at which exploration decreases
        self.update_target_freq = 5  # How often to update target network (in episodes)

        # Memory for experience replay
        self.memory = deque(maxlen=10000)  # Store experiences for replay
        self.batch_size = 64  # Number of experiences to sample per learning step

        # Neural Networks
        self.model = self._build_model()  # Primary network (updated each step)
        self.target_model = self._build_model()  # Target network (updated less frequently)
        self.update_target_model()  # Initialize target with same weights

        # Training metrics
        self.loss_history = []

    def _build_model(self):
        """Create the deep neural network that will approximate Q-values"""
        model = keras.Sequential([
            # Convolutional layers to process the game grid
            keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                                input_shape=self.state_shape),
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),

            # Flatten the 2D feature maps to 1D
            keras.layers.Flatten(),

            # Fully connected layers for decision making
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')  # Q-values for each action
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')  # Mean squared error loss

        return model

    def update_target_model(self):
        """Copy weights from the main model to the target model"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory for replay"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Choose an action based on the current state"""
        # During training, use epsilon-greedy policy (exploration vs exploitation)
        if training and np.random.rand() <= self.epsilon:
            # Exploration: choose random action
            return random.randrange(self.action_size)

        # Exploitation: choose best action according to model
        # Add batch dimension to state
        state_tensor = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state_tensor, verbose=0)

        # Return action with highest predicted Q-value
        return np.argmax(act_values[0])

    def replay(self):
        """Train the model using random samples from memory"""
        # Need enough experiences to sample a batch
        if len(self.memory) < self.batch_size:
            return 0

        # Sample random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Arrays to hold states and next_states
        states = np.zeros((self.batch_size,) + self.state_shape)
        next_states = np.zeros((self.batch_size,) + self.state_shape)

        # Extract states and next_states for batch prediction
        for i, (state, _, _, next_state, _) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state

        # Predict Q-values for current states and next states
        q_values = self.model.predict(states, verbose=0)
        q_values_next = self.target_model.predict(next_states, verbose=0)

        # Create training input and target output arrays
        X = states
        y = q_values

        # Update Q-values for actions taken
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                # If episode is done, there's no future reward
                target = reward
            else:
                # Bellman equation: Q(s,a) = r + γ * max(Q(s',a'))
                target = reward + self.gamma * np.amax(q_values_next[i])

            # Update the Q-value for the action taken
            y[i][action] = target

        # Train the model on the batch
        hist = self.model.fit(X, y, epochs=1, verbose=0)
        self.loss_history.append(hist.history['loss'][0])

        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return hist.history['loss'][0]

    def load(self, name):
        """Load model weights from file"""
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        """Save model weights to file"""
        self.model.save_weights(name)