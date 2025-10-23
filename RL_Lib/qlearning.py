from .base_agent import BaseAgent

class QLearning(BaseAgent):
    """
    Q-Learning algorithm implementation.
    """

    def training(self, episodes=1000, max_steps=200, verbose=True):
        """Train the Q-Learning agent."""
        for ep in range(1, episodes + 1):
            state = self._reset_env()
            self._init_state(state)
            total_reward = 0
            done = False

            for step in range(max_steps):
                # choose action (use current epsilon)
                action = self.action(state)

                next_state, reward, done, info = self._step_env(action)
                self._init_state(next_state)

                s_key = self._get_state_key(state)
                n_key = self._get_state_key(next_state)

                # Q-learning update: Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
                best_next_action = max(self.q_table[n_key], key=self.q_table[n_key].get)
                td_target = reward + self.gamma * self.q_table[n_key][best_next_action]
                td_error = td_target - self.q_table[s_key][action]
                self.q_table[s_key][action] += self.alpha * td_error

                state = next_state
                total_reward += reward

                if done:
                    break

            if verbose and (ep % max(1, episodes//10) == 0 or ep == 1):
                print(f"Episode {ep}/{episodes} - Total Reward: {total_reward}")
