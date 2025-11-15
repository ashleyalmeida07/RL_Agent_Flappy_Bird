# (Agent Logic + Training)
import torch
import torch.nn.functional as F
from model import DQN, ReplayMemory
import random

LR = 0.0005
GAMMA = 0.99

class Agent:
    def __init__(self, state_dim=180, action_dim=2):
        self.policy_net = DQN(state_dim, 256, action_dim)
        self.target_net = DQN(state_dim, 256, action_dim)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory()

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)

        s = torch.tensor(state, dtype=torch.float32)
        q = self.policy_net(s)
        return torch.argmax(q).item()

    def train_step(self):
        if not self.memory.can_sample():
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.target_net(next_states).max(1)[0]
        targets = rewards + GAMMA * next_q * (1 - dones)

        loss = F.mse_loss(q_values, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
