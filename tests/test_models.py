"""Test RL models"""

import torch
import numpy as np
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure dqn module exists
try:
    from src.models.dqn import DQNAgent, QNetwork, ReplayBuffer
except ImportError:
    print("Error: Could not import DQN modules.")
    print("Make sure 'src/models/dqn.py' exists and is correct.")
    sys.exit(1)


class TestModels(unittest.TestCase):

    def test_q_network(self):
        """Test Q-Network"""
        print("Testing Q-Network...")
        
        state_dim = 20
        action_dim = 4
        batch_size = 32
        
        network = QNetwork(state_dim, action_dim, hidden_layers=[64, 64])
        state = torch.randn(batch_size, state_dim)
        q_values = network(state)
        
        self.assertEqual(q_values.shape, (batch_size, action_dim))
        print("✓ Q-Network test passed")

    def test_replay_buffer(self):
        """Test replay buffer"""
        print("\nTesting replay buffer...")
        
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(50):
            state = np.random.randn(10)
            action = np.random.randint(4)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = False
            buffer.push(state, action, reward, next_state, done)
        
        self.assertEqual(len(buffer), 50)
        
        batch = buffer.sample(32)
        states, actions, rewards, next_states, dones = batch
        
        self.assertEqual(states.shape, (32, 10))
        self.assertEqual(actions.shape, (32, 1))
        print("✓ Replay buffer test passed")

    def test_dqn_agent(self):
        """Test DQN agent"""
        print("\nTesting DQN agent...")
        
        state_dim = 20
        action_dim = 4
        
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=[64, 64],
            buffer_size=1000,
            batch_size=32
        )
        
        state = np.random.randn(state_dim)
        action = agent.select_action(state, eval_mode=False)
        self.assertIn(action, range(action_dim))
        
        for _ in range(100):
            agent.store_transition(state, action, 1.0, state, False)
        
        loss = agent.train_step()
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)
        print("✓ DQN agent test passed")

    def test_save_load(self):
        """Test model save/load"""
        print("\nTesting save/load...")
        
        agent = DQNAgent(state_dim=10, action_dim=4)
        save_path = Path('test_model.pth')
        
        agent.save(save_path)
        self.assertTrue(save_path.exists())
        
        agent2 = DQNAgent(state_dim=10, action_dim=4)
        agent2.load(save_path)
        
        state = np.random.randn(10)
        action1 = agent.select_action(state, eval_mode=True)
        action2 = agent2.select_action(state, eval_mode=True)
        
        self.assertEqual(action1, action2)
        
        save_path.unlink()
        print("✓ Save/load test passed")

if __name__ == "__main__":
    unittest.main()