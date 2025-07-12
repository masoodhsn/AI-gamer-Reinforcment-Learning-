import gymnasium as gym
from time import sleep
from Qlearning import QLearning
import numpy as np
#from numpy import loadtxt


env = gym.make("Taxi-v3",render_mode='human').env   #ansi


qlearn = QLearning(env=env, alpha=.1, gamma=.1, epsilon=.5, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
q_table, _ = qlearn.train('./q-table-taxi-driver-1.csv')


(state, _) = env.reset()
epochos, penalties, reward = 0, 0, 0
done = False

while (not done) and (epochos < 100):
    action = np.argmax(q_table[state])
    state, reward, done, t, _ = env.step(action)

    if reward == -10:
        penalties += 1

    print({
        'state': state,
        'action': action,
        'reward': reward
    })
    epochos += 1
    sleep(0.2)

print("Timestep: {}".format(epochos))
print("Penalties: {}".format(penalties))