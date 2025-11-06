"""
Test Suite for Fairness-Aware Traffic Signal Control
tests/test_*.py
"""

# ============================================
# tests/test_reward.py
# ============================================
"""Test reward function"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.reward import FairnessAwareReward, RewardConfig, BaselineReward


def test_reward_basic():
    """Test basic reward computation"""
    print("Testing basic reward computation...")
    
    config = RewardConfig()
    reward_fn = FairnessAwareReward(config)
    
    state_info = {
        'waiting_times': [10, 20, 30, 40],
        'queue_lengths': [2, 3, 1, 2],
        'vehicle_ids': ['v1', 'v2', 'v3', 'v4']
    }
    
    reward, info = reward_fn.compute_reward(state_info)
    
    assert isinstance(reward, float)
    assert 'reward_total' in info
    assert 'reward_efficiency' in info
    assert 'reward_fairness' in info
    assert 'reward_penalty' in info
    
    print(f"  Reward: {reward:.3f}")
    print(f"  Components: {info['reward_efficiency']:.3f}, {info['reward_fairness']:.3f}, {info['reward_penalty']:.3f}")
    print("✓ Basic reward test passed")


def test_reward_starvation():
    """Test starvation penalty"""
    print("\nTesting starvation penalty...")
    
    config = RewardConfig(threshold=100)
    reward_fn = FairnessAwareReward(config)
    
    # Case 1: No starvation
    state_info = {
        'waiting_times': [10, 20, 30, 40],
        'queue_lengths': [2, 3, 1, 2],
        'vehicle_ids': ['v1', 'v2', 'v3', 'v4']
    }
    
    reward1, info1 = reward_fn.compute_reward(state_info)
    penalty1 = info1['reward_penalty']
    
    # Case 2: With starvation
    state_info = {
        'waiting_times': [10, 20, 150, 200],  # Two vehicles starving
        'queue_lengths': [2, 3, 1, 2],
        'vehicle_ids': ['v1', 'v2', 'v3', 'v4']
    }
    
    reward2, info2 = reward_fn.compute_reward(state_info)
    penalty2 = info2['reward_penalty']
    
    print(f"  No starvation penalty: {penalty1:.3f}")
    print(f"  With starvation penalty: {penalty2:.3f}")
    assert penalty2 < penalty1, "Starvation should increase penalty"
    print("✓ Starvation penalty test passed")


def test_reward_fairness():
    """Test fairness component"""
    print("\nTesting fairness component...")
    
    config = RewardConfig()
    reward_fn = FairnessAwareReward(config)
    
    # Case 1: Low variance (fair)
    state_info = {
        'waiting_times': [25, 27, 28, 30],  # Similar waiting times
        'queue_lengths': [2, 2, 2, 2],
        'vehicle_ids': ['v1', 'v2', 'v3', 'v4']
    }
    
    reward1, info1 = reward_fn.compute_reward(state_info)
    fair1 = info1['reward_fairness']
    
    # Case 2: High variance (unfair)
    state_info = {
        'waiting_times': [5, 10, 50, 90],  # Very different waiting times
        'queue_lengths': [2, 2, 2, 2],
        'vehicle_ids': ['v1', 'v2', 'v3', 'v4']
    }
    
    reward2, info2 = reward_fn.compute_reward(state_info)
    fair2 = info2['reward_fairness']
    
    print(f"  Fair distribution: {fair1:.3f} (std={info1['std_waiting_time']:.1f})")
    print(f"  Unfair distribution: {fair2:.3f} (std={info2['std_waiting_time']:.1f})")
    assert fair2 < fair1, "Higher variance should give lower fairness reward"
    print("✓ Fairness component test passed")


def test_episode_metrics():
    """Test episode metrics computation"""
    print("\nTesting episode metrics...")
    
    config = RewardConfig()
    reward_fn = FairnessAwareReward(config)
    
    # Simulate multiple steps
    for _ in range(5):
        state_info = {
            'waiting_times': np.random.uniform(10, 50, 10).tolist(),
            'queue_lengths': np.random.randint(0, 5, 5).tolist(),
            'vehicle_ids': [f'v{i}' for i in range(10)]
        }
        reward_fn.compute_reward(state_info)
    
    metrics = reward_fn.get_episode_metrics()
    
    assert 'mean_waiting_time' in metrics
    assert 'gini_coefficient' in metrics
    assert 'jain_fairness_index' in metrics
    
    print(f"  Metrics: {list(metrics.keys())[:5]}...")
    print("✓ Episode metrics test passed")


def test_baseline_reward():
    """Test baseline reward"""
    print("\nTesting baseline reward...")
    
    reward_fn = BaselineReward()
    
    state_info = {
        'waiting_times': [10, 20, 30, 40],
        'queue_lengths': [2, 3, 1, 2],
        'vehicle_ids': ['v1', 'v2', 'v3', 'v4']
    }
    
    reward, info = reward_fn.compute_reward(state_info)
    
    assert isinstance(reward, float)
    assert reward < 0  # Negative waiting time
    print(f"  Baseline reward: {reward:.3f}")
    print("✓ Baseline reward test passed")


if __name__ == "__main__":
    print("="*60)
    print("Reward Function Test Suite")
    print("="*60)
    
    test_reward_basic()
    test_reward_starvation()
    test_reward_fairness()
    test_episode_metrics()
    test_baseline_reward()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


# ============================================
# tests/test_models.py
# ============================================
"""Test RL models"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dqn import DQNAgent, QNetwork, ReplayBuffer


def test_q_network():
    """Test Q-Network"""
    print("Testing Q-Network...")
    
    state_dim = 20
    action_dim = 4
    batch_size = 32
    
    network = QNetwork(state_dim, action_dim, hidden_layers=[64, 64])
    
    # Forward pass
    state = torch.randn(batch_size, state_dim)
    q_values = network(state)
    
    assert q_values.shape == (batch_size, action_dim)
    print(f"  Q-values shape: {q_values.shape}")
    print("✓ Q-Network test passed")


def test_replay_buffer():
    """Test replay buffer"""
    print("\nTesting replay buffer...")
    
    buffer = ReplayBuffer(capacity=100)
    
    # Add experiences
    for i in range(50):
        state = np.random.randn(10)
        action = np.random.randint(4)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
    
    assert len(buffer) == 50
    
    # Sample batch
    batch = buffer.sample(32)
    states, actions, rewards, next_states, dones = batch
    
    assert states.shape == (32, 10)
    assert actions.shape == (32,)
    print(f"  Buffer size: {len(buffer)}")
    print(f"  Batch shapes: states={states.shape}, actions={actions.shape}")
    print("✓ Replay buffer test passed")


def test_dqn_agent():
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
    
    # Test action selection
    state = np.random.randn(state_dim)
    action = agent.select_action(state, eval_mode=False)
    assert 0 <= action < action_dim
    
    # Store transitions
    for _ in range(100):
        state = np.random.randn(state_dim)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = False
        
        agent.store_transition(state, action, reward, next_state, done)
    
    # Train
    loss = agent.train_step()
    assert loss is not None and loss >= 0
    
    print(f"  Agent parameters: {agent.count_parameters():,}")
    print(f"  Training loss: {loss:.4f}")
    print("✓ DQN agent test passed")


def test_save_load():
    """Test model save/load"""
    print("\nTesting save/load...")
    
    agent = DQNAgent(state_dim=10, action_dim=4)
    
    # Save
    save_path = Path('test_model.pth')
    agent.save(save_path)
    
    # Load
    agent2 = DQNAgent(state_dim=10, action_dim=4)
    agent2.load(save_path)
    
    # Compare
    state = np.random.randn(10)
    action1 = agent.select_action(state, eval_mode=True)
    action2 = agent2.select_action(state, eval_mode=True)
    
    assert action1 == action2, "Loaded model should produce same actions"
    
    # Cleanup
    save_path.unlink()
    
    print("✓ Save/load test passed")


if __name__ == "__main__":
    print("="*60)
    print("Model Test Suite")
    print("="*60)
    
    test_q_network()
    test_replay_buffer()
    test_dqn_agent()
    test_save_load()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


# ============================================
# tests/run_all_tests.py
# ============================================
"""Run all tests"""

import sys
from pathlib import Path

print("="*60)
print("Running All Tests")
print("="*60)

# Run reward tests
print("\n1. Reward Function Tests:")
print("-" * 40)
import test_reward
test_reward.test_reward_basic()
test_reward.test_reward_starvation()
test_reward.test_reward_fairness()
test_reward.test_episode_metrics()
test_reward.test_baseline_reward()

# Run model tests
print("\n2. Model Tests:")
print("-" * 40)
import test_models
test_models.test_q_network()
test_models.test_replay_buffer()
test_models.test_dqn_agent()
test_models.test_save_load()

print("\n" + "="*60)
print("ALL TESTS PASSED! ✓")
print("="*60)
