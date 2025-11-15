# Train the Agent
import gymnasium as gym
import flappy_bird_gymnasium
from agent import Agent
import torch

EPISODES = 1000 #10,000

env = gym.make("FlappyBird-v0", render_mode=None)

# Auto-detect correct state dimension (180 on your system)
state, _ = env.reset()
state_dim = len(state)
print("Detected state dimension:", state_dim)

agent = Agent(state_dim=state_dim)

scores = []

for ep in range(EPISODES):
    state, _ = env.reset()
    done = False
    score = 0

    while not done:
        action = agent.select_action(state)

        next_state, reward, done, truncated, info = env.step(action)

        agent.memory.push((state, action, reward, next_state, done))
        agent.train_step()

        state = next_state
        score += reward

    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    if ep % 20 == 0:
        agent.update_target()

    print(f"Episode {ep}   Score: {score:.2f}   Epsilon: {agent.epsilon:.3f}")

torch.save(agent.policy_net.state_dict(), "flappy_dqn_ram.pth")
print("Training finished and saved!")
