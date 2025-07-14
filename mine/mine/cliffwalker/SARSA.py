import numpy as np
import random
from BlockGridAnimator import BlockGridAnimator as bga  

class Sarsa:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.data = []
    
    def select_action(self, state):
        randomNumber = random.uniform(0, 1)
        if randomNumber < self.epsilon:
            return self.env.action_space.sample() # Explore
        return np.argmax(self.q_table[state]) # Exploit
    
    def train(self):
        actions_per_episode = []
        for i in range(1, self.episodes + 1):
            (state, _) = self.env.reset()
            reward = 0
            done = False
            actions = 0
            rewards = 0

            action = self.select_action(state)
            self.data.append(self.q_table.copy())
            
            while not done:

                next_state, reward, done, truncated, terminal = self.env.step(action)
                if reward == -100 : done = True
                if next_state == 47: reward = 100

                old_value = self.q_table[state, action]
                next_action = self.select_action(next_state)
                
                new_value = old_value + self.alpha * (reward + self.gamma * self.q_table[next_state, next_action] - old_value)     # differ from QLearning
                self.q_table[state, action] = new_value

                state = next_state
                action = next_action

                actions += 1
                rewards += reward

            actions_per_episode.append(rewards)
            if i % 100 == 0:
                print("Episodes: " + str(i) + "\n")

            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec

        
        data = np.array(self.data)


        data_reshape = np.zeros((len(data), 4, 12, 4))
        for i in range(len(data)):
            rows = np.zeros((4, 12, 4))
            col = np.zeros((12, 4))
            for j in range(len(data[i])):
                col[j % 12] = data[i,j]
                if j % 12 == 11:
                    rows[j // 12] = col
                    col = np.zeros((12, 4))
                

            data_reshape[i] = rows
        print(data_reshape.shape)
        anim = bga(data_reshape)
        anim.show(500)                # To preview animation live
        #anim.save("grid_output.gif")  # To save as GIF


        return self.q_table, actions_per_episode