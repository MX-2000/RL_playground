import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")  # Make plotting style similar to R

SIZE = 10  # Grid Size
EPISODES = 125_000
MOVE_PENALTY = 1e1
ENEMY_PENALTY = 1e5
FOOD_REWARD = 1e2
NUM_STEPS = 200  # Max num of steps to get to a food or enemy

EPSILON_DECAY = 1e-3
EPSILON_START = 1
EPSILON_MIN = 0
epsilon = EPSILON_START

SHOW_EVERY = 3000  # How often do we wanna show the resuts

start_q_table = (
    "qtable-1721213795.pickle"  # to load an existing q table and train from a point
)

LEARNING_RATE = 1e-1
DISCOUNT = 0.5

# Representation of player, good & enemy
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

N_ACTIONS = 4

# Blobs colors in BGR
d = {1: (255, 255, 255), 2: (0, 255, 0), 3: (0, 0, 255)}


class Blob:
    def __init__(self, other_blobs: list) -> None:
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

        while self.overlaps(other_blobs):
            self.x = np.random.randint(0, SIZE)
            self.y = np.random.randint(0, SIZE)

    def overlaps(self, other_blobs):
        for blob in other_blobs:
            if self.x == blob.x and self.y == blob.y:
                return True
        return False

    def __str__(self):
        return f"{self.y},{self.x}"

    def __sub__(self, other_blob):
        return (self.x - other_blob.x, self.y - other_blob.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=0)
        elif choice == 1:
            self.move(x=-1, y=0)
        elif choice == 2:
            self.move(x=0, y=1)
        elif choice == 3:
            self.move(x=0, y=-1)

    def move(self, x=False, y=False):
        """
        If no x or y is passed, we make it move randomly
        """
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # Handles wall issues
        if self.x < 0:
            self.x = 0
        if self.x > SIZE - 1:
            self.x = SIZE - 1

        if self.y < 0:
            self.y = 0
        if self.y > SIZE - 1:
            self.y = SIZE - 1


if start_q_table is None:
    # q shape is x,y,x,y,n_actions because we're taking the relative x and y from player_to_food and player_to_enemy so we remove one x,y in the q_learning shape
    q_table = {}
    for x1 in range(-SIZE + 1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    # Each entry has 4 values for 4 potential actions
                    # We initialize with random values between -5 and 0
                    q_table[((x1, y1), (x2, y2))] = [
                        np.random.uniform(-50, 50) for i in range(N_ACTIONS)
                    ]
else:
    # if one q_table pretrained we can load it
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []

for episode in range(EPISODES):
    player = Blob([])
    food = Blob([player])
    enemy = Blob([player, food])

    # print([f"{blob}" for blob in [player, food, enemy]])

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(NUM_STEPS):

        # Select action based on epsilon policy
        obs = (player - food, player - enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, N_ACTIONS)

        player.action(action)

        # Reward part
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        # Q learning update
        new_obs = (player - food, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q
            )

        q_table[obs][action] = new_q

        if show:
            # The rep is a squared grid of RGB so shape is (SIZE,SIZE,3)
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((500, 500), resample=Image.BOX)
            cv2.imshow("", np.array(img))

            # If we end the episode we want to pause a bit to visualize it - shady code
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    # Append that reward to the episode rewards
    episode_rewards.append(episode_reward)

    # Decay epsilon
    # epsilon *= EPSILON_DECAY

    epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * np.exp(
        -EPSILON_DECAY * episode
    )

moving_avg = np.convolve(
    episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid"
)
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel(f"Episode number")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
