"""
DQN Agent Implementation
src/models/dqn.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from pathlib import Path

# Experience tuple
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'reward', 'next_state', 'done')
)

class QNetwork(nn.Module):
    """Deep Q-Network (MLP)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list = [256, 256]):
        super(QNetwork, self).__init__()
        
        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Fixed-size replay buffer to store experience tuples"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Save an experience"""
        e = Experience(state, action, reward, next_state, done)
        self.buffer.append(e)
        
    def sample(self, batch_size: int) -> tuple:
        """Randomly sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.tensor(np.vstack([e.state for e in batch]), dtype=torch.float32)
        actions = torch.tensor(np.vstack([e.action for e in batch]), dtype=torch.int64)
        rewards = torch.tensor(np.vstack([e.reward for e in batch]), dtype=torch.float32)
        next_states = torch.tensor(np.vstack([e.next_state for e in batch]), dtype=torch.float32)
        dones = torch.tensor(np.vstack([e.done for e in batch]), dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent
    Interacts with and learns from the environment.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list = [256, 256],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 128,
        device: torch.device = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = ReplayBuffer(buffer_size)
        
        self.steps_done = 0
        
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Selects an action using epsilon-greedy policy"""
        
        # Epsilon-greedy
        if not eval_mode and random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # Greedy
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
            
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.push(state, action, reward, next_state, done)
        
    def train_step(self) -> float:
        """Perform one step of training"""
        if len(self.memory) < self.batch_size:
            return 0.0 # Not enough samples
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Get Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)
        
        # Get V(s')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
        # Compute target Q(s, a)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss (Huber loss)
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps_done += 1
        return loss.item()
        
    def update_target_network(self):
        """Update target network with policy network's weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def save(self, filepath: Path):
        """Save model weights"""
        torch.save(self.policy_net.state_dict(), filepath)
        
    def load(self, filepath: Path):
        """Load model weights"""
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
