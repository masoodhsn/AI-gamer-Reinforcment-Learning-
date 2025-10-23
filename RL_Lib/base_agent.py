import numpy as np
import pickle
import os

class BaseAgent:
    """
    Base class for RL agents (SARSA, Q-Learning, etc.)
    Provides common methods for model management and action selection.
    Assumes discrete action space (env.action_space.n).
    """

    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # maps state_key -> {action: value}

        # ensure discrete action space
        if not hasattr(self.env.action_space, "n"):
            raise ValueError("This library currently supports only discrete action spaces (env.action_space.n).")

    # -------------------------
    # Helpers for env compatibility
    # -------------------------
    def _reset_env(self):
        """
        Call env.reset() and return the observation (works with both gym and gymnasium signatures).
        """
        res = self.env.reset()
        # gymnasium: (obs, info) ; older gym: obs
        if isinstance(res, tuple) and len(res) >= 1:
            return res[0]
        return res

    def _step_env(self, action):
        """
        Call env.step(action) and return (obs, reward, done, info) for both gym and gymnasium.
        """
        res = self.env.step(action)
        if len(res) == 5:
            # gymnasium: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = res
            done = terminated or truncated
            return obs, reward, done, info
        elif len(res) == 4:
            # gym: obs, reward, done, info
            obs, reward, done, info = res
            return obs, reward, done, info
        else:
            raise ValueError("Unrecognized env.step() return signature: length = {}".format(len(res)))

    # -------------------------
    # State -> hashable key conversion
    # -------------------------
    def _to_hashable(self, x):
        """
        Convert various observation types (dict, list, tuple, np.ndarray, scalars) into a hashable structure.
        Returns a nested tuple representation.
        """
        if isinstance(x, dict):
            # sort keys to make deterministic
            return tuple((k, self._to_hashable(x[k])) for k in sorted(x.keys()))
        if isinstance(x, (list, tuple, np.ndarray)):
            # convert numpy arrays to list first to handle nested arrays
            arr = np.array(x)
            try:
                pylist = arr.tolist()
            except Exception:
                # fallback: convert elements individually
                return tuple(self._to_hashable(el) for el in x)
            return tuple(self._to_hashable(el) for el in pylist)
        if isinstance(x, (int, float, str, bool, type(None))):
            return x
        # fallback: convert to string representation (not ideal but safe)
        return str(x)

    def _get_state_key(self, state):
        """Return a hashable key for a given observation/state."""
        return self._to_hashable(state)

    # -------------------------
    # Q-table management
    # -------------------------
    def _init_state(self, state):
        """Initialize Q-values for a new state (if not present)."""
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            # create action entries 0..n-1
            n_actions = self.env.action_space.n
            self.q_table[state_key] = {a: 0.0 for a in range(n_actions)}

    def model_load(self, filename):
        """Load a trained model (Q-table) from a file."""
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"Model loaded from {filename}")
        else:
            print(f"Model file not found: {filename}")

    def save_model(self, filename):
        """Save the current model (Q-table) to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {filename}")

    # -------------------------
    # Policy / Action selection
    # -------------------------
    def action(self, state, deterministic=False):
        """
        Choose an action for the given state using epsilon-greedy policy.
        If deterministic=True, always choose best action (no epsilon).
        """
        state_key = self._get_state_key(state)
        # if we haven't seen this state, either return random (explore) or init it
        if state_key not in self.q_table:
            # choose random action as fallback
            return self.env.action_space.sample()

        if (not deterministic) and (np.random.rand() < self.epsilon):
            return self.env.action_space.sample()

        # choose best action (ties broken deterministically)
        action_values = self.q_table[state_key]
        # find action with max value (if multiple, pick smallest action index)
        best_action = max(sorted(action_values.keys()), key=lambda a: action_values[a])
        return best_action
