import numpy as np
import os
from gymnasium.spaces import Discrete
import gymnasium as gym
import time

class QLearning:
    def __init__(self, env_name, learning_rate=0.1, gamma=0.99, epsilon=0.1,
                 total_timesteps=20000, policy_dir="policies"):
        """
        env_name: Gymnasium environment name (string)
        """
        self.env_name = env_name
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_dir = policy_dir
        os.makedirs(policy_dir, exist_ok=True)
        self.policy_file = os.path.join(policy_dir, f"QLearning_{env_name}.npy")

        # Load or train policy
        if os.path.exists(self.policy_file):
            print(f"Loading existing Q-table from {self.policy_file}")
            self.q_table = np.load(self.policy_file)
        else:
            print("No existing policy found. Training headless...")
            self._train(total_timesteps)
            np.save(self.policy_file, self.q_table)
            print(f"Training finished. Policy saved to {self.policy_file}")

    def _train(self, total_timesteps):
        # Headless training environment
        env = gym.make(self.env_name)
        assert isinstance(env.observation_space, Discrete), "Env must have discrete states"
        assert isinstance(env.action_space, Discrete), "Env must have discrete actions"

        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

        obs, _ = env.reset()
        for _ in range(total_timesteps):
            if np.random.rand() < self.epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(self.q_table[obs])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next = np.max(self.q_table[next_obs])
            td_target = reward + self.gamma * best_next * (1 - done)
            self.q_table[obs, action] += self.lr * (td_target - self.q_table[obs, action])

            obs = next_obs if not done else env.reset()[0]
        env.close()

    def predict(self, obs, deterministic=True):
        action = np.argmax(self.q_table[obs])
        return action, None

    def play(self, render_delay=0.3):
        """
        Play one episode with graphical rendering.
        """
        env = gym.make(self.env_name, render_mode="human")
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = self.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()
            import time
            time.sleep(render_delay)
        print(f"Episode finished. Total reward: {total_reward}")
        env.close()



class SARSA:
    def __init__(self, env_name, learning_rate=0.1, gamma=0.99, epsilon=0.1,
                 total_timesteps=20000, policy_dir="policies"):
        """
        env_name: Gymnasium environment name (string)
        """
        self.env_name = env_name
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_dir = policy_dir
        os.makedirs(policy_dir, exist_ok=True)
        self.policy_file = os.path.join(policy_dir, f"SARSA_{env_name}.npy")

        # Load existing policy if exists, otherwise train
        if os.path.exists(self.policy_file):
            print(f"Loading existing Q-table from {self.policy_file}")
            self.q_table = np.load(self.policy_file)
        else:
            print("No existing policy found. Training headless...")
            self._train(total_timesteps)
            np.save(self.policy_file, self.q_table)
            print(f"Training finished. Policy saved to {self.policy_file}")

    def _epsilon_greedy(self, obs, env):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        return np.argmax(self.q_table[obs])

    def _train(self, total_timesteps):
        # Headless environment
        env = gym.make(self.env_name)
        assert isinstance(env.observation_space, Discrete), "Env must have discrete states"
        assert isinstance(env.action_space, Discrete), "Env must have discrete actions"

        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

        obs, _ = env.reset()
        action = self._epsilon_greedy(obs, env)

        for _ in range(total_timesteps):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = self._epsilon_greedy(next_obs, env) if not done else None

            td_target = reward + self.gamma * (self.q_table[next_obs, next_action] if not done else 0)
            self.q_table[obs, action] += self.lr * (td_target - self.q_table[obs, action])

            if done:
                obs, _ = env.reset()
                action = self._epsilon_greedy(obs, env)
            else:
                obs, action = next_obs, next_action

        env.close()

    def predict(self, obs, deterministic=True):
        """Return greedy action based on learned Q-table."""
        action = np.argmax(self.q_table[obs])
        return action, None

    def play(self, render_delay=0.3):
        """
        Play one episode with graphical rendering.
        """
        env = gym.make(self.env_name, render_mode="human")
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = self.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()
            time.sleep(render_delay)

        print(f"Episode finished. Total reward: {total_reward}")
        env.close()
