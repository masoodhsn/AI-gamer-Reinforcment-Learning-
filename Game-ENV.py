from RL_AI import QLearning, SARSA

# --- Q-Learning ---
q_agent = QLearning("CliffWalking-v0", total_timesteps=200000)
q_agent.play(render_delay=0.3)

# --- SARSA ---
sarsa_agent = SARSA("CliffWalking-v0", total_timesteps=200000)
sarsa_agent.play(render_delay=0.3)
