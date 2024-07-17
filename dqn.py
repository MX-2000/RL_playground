from collections import deque
import time
import random
import os
from tqdm import tqdm

import numpy as np

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from blob import BlobEnv


LEARNING_RATE = 1e-3
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 10_000
MINIBATCH_SIZE = 64
MODEL_NAME = "256x2"
DISCOUNT = 0.99  # gamma discount factor
UPDATE_TARGET_FREQUENCY = 5  # how often do we update our target network
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

env = BlobEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir("models"):
    os.makedirs("models")


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=step)
            self.writer.flush()


class DQNAgent:
    def __init__(self):

        # this is what we trained
        self.model = self.create_model()
        # This one is fixed and updated every x, but this is the one we take action on
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"log_dir/{MODEL_NAME}-{int(time.time())}"
        )

        self.target_update_counter = (
            0  # Keeping track of when to update the target network
        )

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=LEARNING_RATE),
            metrics=["accuracy"],
        )
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        # should only add a dimension 1 at the beginning. convert (a,b,c) to (1,a,b,c), to mean a batch of 1 example
        # We normalize by 255 because we're passing an image
        # We should get back a list for the batches. Here we only care about the first element
        return self.model.predict(
            np.array(state).reshape(-1, *state.shape) / 255, verbose=0
        )[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = (
            np.array([transition[0] for transition in minibatch]) / 255
        )  # Current states in each minibatch example

        current_qs_list = self.model.predict(
            current_states, verbose=0
        )  # Predict from the constantly trained model

        new_current_states = (
            np.array([transition[3] for transition in minibatch]) / 255
        )  # Get the next step for each minibatch example
        future_qs_list = self.target_model.predict(
            new_current_states, verbose=0
        )  # Predict from the target

        X = []
        y = []

        for index, (
            current_state,
            action,
            reward,
            new_current_state,
            done,
        ) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Updating our q values
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # We now have our new input and output values ready to fit
            X.append(current_state)
            y.append(current_qs)

        # Tensorboard only in terminal state
        self.model.fit(
            np.array(X) / 255,
            np.array(y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if terminal_state else None,
        )

        # Do we want to copy the weights and update the target network?
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_FREQUENCY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


agent = DQNAgent()

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episode"):
    agent.tensorboard.step = episode

    episode_reward = 0
    step = 1
    current_state = env.reset()

    done = False

    while not done:
        if np.random.random() < epsilon:
            action = np.argmax(agent.get_qs(current_state, step))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(
            ep_rewards[-AGGREGATE_STATS_EVERY:]
        )
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(
            reward_avg=average_reward,
            reward_min=min_reward,
            reward_max=max_reward,
            epsilon=epsilon,
        )

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(
                f"models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model"
            )

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
