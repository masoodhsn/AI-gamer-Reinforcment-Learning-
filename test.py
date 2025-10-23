from RL_Lib import QLearning, SARSA
import gymnasium as gym
import time

# ساخت محیط Cliff Walking
env = gym.make("CliffWalking-v0", render_mode="anis")  # ansi = text mode render

# انتخاب الگوریتم (می‌تونی SARSA هم تست کنی)
agent = SARSA(env, alpha=0.1, gamma=0.99, epsilon=0.1)

# آموزش مدل
agent.training(episodes=2000, max_steps=200)
agent.save_model("cliff_q_model.pkl")

# بارگذاری مدل آموزش‌دیده
env = gym.make("CliffWalking-v0", render_mode="human") 
agent = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1)
agent.model_load("cliff_q_model.pkl")
# تست مدل آموزش‌دیده به‌صورت بصری (متنی)
num_episodes = 2
for ep in range(num_episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    total_reward = 0
    done = False

    print(f"\n🎮 Episode {ep + 1} starting...")
    #print(env.render(), end='\r')

    for step in range(100):
        # انتخاب عمل با مدل آموزش‌دیده
        action = agent.action(state, deterministic=True)

        next_state, reward, done, info = agent._step_env(action)
        total_reward += reward
        state = next_state

        # نمایش وضعیت محیط در هر گام
        #print(env.render(),end='\r')
        time.sleep(0.2)

        if done:
            print(f"✅ Episode {ep + 1} finished. Total reward: {total_reward}")
            break

env.close()
