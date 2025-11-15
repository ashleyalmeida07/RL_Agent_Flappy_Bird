# (Watch the AI Play — REAL GAME WINDOW)
import gymnasium as gym
import flappy_bird_gymnasium
import torch
from agent import Agent

env = gym.make("FlappyBird-v0", render_mode="human")

state, _ = env.reset()
state_dim = len(state)

agent = Agent(state_dim=state_dim)
agent.policy_net.load_state_dict(torch.load("flappy_dqn_ram.pth"))

while True:
    s = torch.tensor(state, dtype=torch.float32)
    q = agent.policy_net(s)
    action = torch.argmax(q).item()

    next_state, reward, done, truncated, info = env.step(action)

    state = next_state

    if done:
        state, _ = env.reset()
