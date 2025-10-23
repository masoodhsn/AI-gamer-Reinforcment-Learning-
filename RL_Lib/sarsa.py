from .base_agent import BaseAgent

class SARSA(BaseAgent):
    """
    SARSA algorithm implementation.
    """

    def training(self, episodes=1000, max_steps=200, verbose=True):
        """Train the SARSA agent."""
        for ep in range(1, episodes + 1):
            state = self._reset_env()
            self._init_state(state)
            action = self.action(state)
            total_reward = 0
            done = False

            for step in range(max_steps):
                next_state, reward, done, info = self._step_env(action)
                self._init_state(next_state)
                next_action = self.action(next_state)

                s_key = self._get_state_key(state)
                n_key = self._get_state_key(next_state)

                # SARSA update: Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))
                td_target = reward + self.gamma * self.q_table[n_key][next_action]
                td_error = td_target - self.q_table[s_key][action]
                self.q_table[s_key][action] += self.alpha * td_error

                state = next_state
                action = next_action
                total_reward += reward

                if done:
                    break

            if verbose and (ep % max(1, episodes//10) == 0 or ep == 1):
                print(f"Episode {ep}/{episodes} - Total Reward: {total_reward}")
