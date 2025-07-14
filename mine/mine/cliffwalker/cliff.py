import gymnasium as gym
from QLearning import QLearning
from SARSA import Sarsa
import numpy as np

env = gym.make("CliffWalking-v0").env  #,render_mode="human").env

# Qlearning
#print("Training using Qlearning")
#qlearning = QLearning(env, 0.9, 0.9, 0.4, 0.05, 0.99, 100)
#q_table, _ = qlearning.train()

# SARSA
print("Training using SARSA")
sarsa = Sarsa(env, 0.9, 0.9, 0.4, 0.05, 0.99, 100)
sarsa_table, _ = sarsa.train()



env = gym.make("CliffWalking-v0", render_mode="human").env

print("Playing ...")
(state, _) = env.reset()
rewards = 0
actions = 0
done = False

while not done:
    #action = np.argmax(q_table[state])
    action = np.argmax(sarsa_table[state])
    state, reward, done, truncated, _ = env.step(action)

    rewards = rewards + reward
    actions += 1
    

# Result
print(f"Actions: {actions}")
print(f"Rewards: {rewards}")

