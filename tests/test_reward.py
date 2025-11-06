"""Test reward function"""

import numpy as np
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.reward import FairnessAwareReward, RewardConfig, BaselineReward

class TestRewardFunction(unittest.TestCase):

    def test_reward_basic(self):
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
        
        self.assertIsInstance(reward, float)
        self.assertIn('reward_total', info)
        self.assertIn('reward_efficiency', info)
        self.assertIn('reward_fairness', info)
        self.assertIn('reward_penalty', info)
        
        print(f"  Reward: {reward:.3f}")
        print("✓ Basic reward test passed")

    def test_reward_starvation(self):
        """Test starvation penalty"""
        print("\nTesting starvation penalty...")
        
        config = RewardConfig(threshold=100)
        reward_fn = FairnessAwareReward(config)
        
        # Case 1: No starvation
        state_info1 = {
            'waiting_times': [10, 20, 30, 40],
            'queue_lengths': [2, 3, 1, 2],
            'vehicle_ids': ['v1', 'v2', 'v3', 'v4']
        }
        reward1, info1 = reward_fn.compute_reward(state_info1)
        
        # Case 2: With starvation
        state_info2 = {
            'waiting_times': [10, 20, 150, 200],  # Two vehicles starving
            'queue_lengths': [2, 3, 1, 2],
            'vehicle_ids': ['v1', 'v2', 'v3', 'v4']
        }
        reward2, info2 = reward_fn.compute_reward(state_info2)
        
        self.assertLess(info2['reward_penalty'], info1['reward_penalty'])
        print("✓ Starvation penalty test passed")

    def test_reward_fairness(self):
        """Test fairness component"""
        print("\nTesting fairness component...")
        
        config = RewardConfig()
        reward_fn = FairnessAwareReward(config)
        
        # Case 1: Low variance (fair)
        state_info1 = {
            'waiting_times': [25, 27, 28, 30],
            'queue_lengths': [2, 2, 2, 2],
            'vehicle_ids': ['v1', 'v2', 'v3', 'v4']
        }
        reward1, info1 = reward_fn.compute_reward(state_info1)
        
        # Case 2: High variance (unfair)
        state_info2 = {
            'waiting_times': [5, 10, 50, 90],
            'queue_lengths': [2, 2, 2, 2],
            'vehicle_ids': ['v1', 'v2', 'v3', 'v4']
        }
        reward2, info2 = reward_fn.compute_reward(state_info2)
        
        self.assertLess(info2['reward_fairness'], info1['reward_fairness'])
        print("✓ Fairness component test passed")

    def test_episode_metrics(self):
        """Test episode metrics computation"""
        print("\nTesting episode metrics...")
        
        config = RewardConfig()
        reward_fn = FairnessAwareReward(config)
        
        for _ in range(5):
            state_info = {
                'waiting_times': np.random.uniform(10, 50, 10).tolist(),
                'queue_lengths': np.random.randint(0, 5, 5).tolist(),
                'vehicle_ids': [f'v{i}' for i in range(10)]
            }
            reward_fn.compute_reward(state_info)
        
        metrics = reward_fn.get_episode_metrics()
        
        self.assertIn('mean_waiting_time', metrics)
        self.assertIn('gini_coefficient', metrics)
        self.assertIn('jain_fairness_index', metrics)
        print("✓ Episode metrics test passed")

    def test_baseline_reward(self):
        """Test baseline reward"""
        print("\nTesting baseline reward...")
        
        reward_fn = BaselineReward()
        state_info = {
            'waiting_times': [10, 20, 30, 40],
            'queue_lengths': [2, 3, 1, 2],
            'vehicle_ids': ['v1', 'v2', 'v3', 'v4']
        }
        reward, info = reward_fn.compute_reward(state_info)
        
        self.assertIsInstance(reward, float)
        self.assertLess(reward, 0)
        print("✓ Baseline reward test passed")

if __name__ == "__main__":
    unittest.main()
