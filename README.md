# RL Agent for Flappy Bird

A reinforcement learning agent that learns to play Flappy Bird using Deep Q-Network (DQN) algorithm.

## Overview

This project implements a DQN-based agent that trains on the Flappy Bird game environment. The agent learns optimal gameplay through experience replay and Q-learning, improving its performance over successive episodes.

## Features

- **Deep Q-Network (DQN)**: Neural network with two hidden layers for Q-value approximation
- **Experience Replay**: Memory buffer storing up to 50,000 experiences for stable training
- **Target Network**: Separate target network updated periodically for training stability
- **Epsilon-Greedy Exploration**: Balanced exploration and exploitation strategy

## Project Structure

```
├── agent.py      # Agent logic with DQN training loop
├── model.py      # Neural network architecture and replay memory
├── train.py      # Training script
└── watch.py      # Watch trained agent play
```

## Requirements

- Python 3.x
- PyTorch
- Gymnasium
- flappy-bird-gymnasium

## Installation

```bash
pip install torch gymnasium flappy-bird-gymnasium
```

## Usage

### Training

Train the agent for 1000 episodes (adjustable in `train.py`):

```bash
python train.py
```

The trained model will be saved as `flappy_dqn_ram.pth`.

### Watch Trained Agent

```bash
python watch.py
```

## Hyperparameters

- **Learning Rate**: 0.0005
- **Discount Factor (γ)**: 0.99
- **Replay Memory Size**: 50,000
- **Batch Size**: 64
- **Epsilon Decay**: 0.995
- **Minimum Epsilon**: 0.01
- **Target Network Update**: Every 20 episodes

## How It Works

1. **State Representation**: 180-dimensional state vector from the Flappy Bird environment
2. **Action Space**: 2 actions (flap or no flap)
3. **Reward System**: Environment provides rewards based on survival and progress
4. **Training**: Agent learns Q-values through temporal difference learning
5. **Exploration**: Epsilon-greedy strategy with decaying exploration rate

## Results

The agent progressively improves its score through training, learning to navigate through pipes more effectively as episodes increase.

