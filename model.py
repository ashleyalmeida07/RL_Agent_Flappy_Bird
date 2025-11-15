# (DQN + Replay Buffer)
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

MEMORY_SIZE = 50000
BATCH_SIZE = 64

class DQN(nn.Module):
    def __init__(self, input_size=180, hidden=256, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)

    def push(self, exp):
        self.memory.append(exp)

    def sample(self):
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(list(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(list(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def can_sample(self):
        return len(self.memory) >= BATCH_SIZE
