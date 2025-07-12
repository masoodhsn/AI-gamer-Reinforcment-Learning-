import gymnasium as gym
from QLearning import QLearning
import numpy as np

env = gym.make("CliffWalking-v0").env  #,render_mode="human").env

# Qlearning
print("Training using Qlearning")
qlearning = QLearning(env, 0.9, 0.9, 0.4, 0.05, 0.99, 100)
q_table, _ = qlearning.train()


env = gym.make("CliffWalking-v0", render_mode="human").env

# QLearning
print("Playing using QLearning")
(state, _) = env.reset()
rewards_q = 0
actions_q = 0
done = False

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, truncated, _ = env.step(action)

    rewards_q = rewards_q + reward
    actions_q += 1
    

# Result
print(f"Actions by Qlearning: {actions_q}")
print(f"Rewards for Qlearning: {rewards_q}")