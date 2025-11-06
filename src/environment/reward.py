"""
Fairness-Aware Reward Function for Traffic Signal Control
src/environment/reward.py
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward function parameters"""
    # Component weights
    w_efficiency: float = 0.4
    w_fairness: float = 0.4
    w_penalty: float = 0.2
    
    # Efficiency parameters
    alpha: float = 1.0  # Waiting time weight
    beta: float = 0.5   # Queue length weight
    
    # Fairness parameters
    gamma: float = 2.0   # Std deviation weight
    delta: float = 1.5   # Max wait weight
    
    # Penalty parameters
    lambda_: float = 0.1      # Exponential penalty rate
    threshold: float = 120.0  # Starvation threshold (seconds)
    
    # Normalization (optional, learned from data)
    normalize: bool = True
    mean_wait: float = 30.0
    std_wait: float = 20.0


class FairnessAwareReward:
    """
    Novel fairness-aware reward function that balances efficiency and equity.
    
    The reward combines three components:
    1. Efficiency: Minimizes average waiting time and queue length
    2. Fairness: Penalizes variance in waiting times
    3. Penalty: Heavily penalizes vehicles waiting exceptionally long
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.episode_stats = {
            'waiting_times': [],
            'queue_lengths': [],
            'starvation_count': 0,
            'max_wait': 0
        }
        
    def reset(self):
        """Reset episode statistics"""
        self.episode_stats = {
            'waiting_times': [],
            'queue_lengths': [],
            'starvation_count': 0,
            'max_wait': 0
        }
    
    def compute_reward(self, state_info: Dict) -> Tuple[float, Dict]:
        """
        Compute the fairness-aware reward.
        
        Args:
            state_info: Dictionary containing:
                - 'waiting_times': List of waiting times for all vehicles
                - 'queue_lengths': List of queue lengths per lane
                - 'vehicle_ids': List of vehicle IDs
                
        Returns:
            reward: Scalar reward value
            info: Dictionary with detailed reward components
        """
        waiting_times = np.array(state_info['waiting_times'])
        queue_lengths = np.array(state_info['queue_lengths'])
        
        # Handle edge case: no vehicles
        if len(waiting_times) == 0:
            return 0.0, self._empty_info()
        
        # 1. EFFICIENCY COMPONENT
        r_efficiency = self._compute_efficiency(waiting_times, queue_lengths)
        
        # 2. FAIRNESS COMPONENT
        r_fairness = self._compute_fairness(waiting_times)
        
        # 3. STARVATION PENALTY
        r_penalty = self._compute_penalty(waiting_times)
        
        # Combine components
        total_reward = (
            self.config.w_efficiency * r_efficiency +
            self.config.w_fairness * r_fairness +
            self.config.w_penalty * r_penalty
        )
        
        # Update episode statistics
        self._update_stats(waiting_times, queue_lengths)
        
        # Detailed information for logging
        info = {
            'reward_total': total_reward,
            'reward_efficiency': r_efficiency,
            'reward_fairness': r_fairness,
            'reward_penalty': r_penalty,
            'avg_waiting_time': np.mean(waiting_times),
            'std_waiting_time': np.std(waiting_times),
            'max_waiting_time': np.max(waiting_times),
            'avg_queue_length': np.mean(queue_lengths),
            'num_vehicles': len(waiting_times),
            'starvation_count': np.sum(waiting_times > self.config.threshold)
        }
        
        return total_reward, info
    
    def _compute_efficiency(self, waiting_times: np.ndarray, 
                           queue_lengths: np.ndarray) -> float:
        """
        Compute efficiency reward component.
        
        R_efficiency = -α * avg_wait - β * avg_queue
        """
        avg_wait = np.mean(waiting_times)
        avg_queue = np.mean(queue_lengths)
        
        # Normalize if enabled
        if self.config.normalize:
            avg_wait = (avg_wait - self.config.mean_wait) / self.config.std_wait
        
        r_eff = -(self.config.alpha * avg_wait + 
                  self.config.beta * avg_queue)
        
        return r_eff
    
    def _compute_fairness(self, waiting_times: np.ndarray) -> float:
        """
        Compute fairness reward component.
        
        R_fairness = -γ * std(wait) - δ * max(wait)
        
        This penalizes both variance (inequality) and extreme cases.
        """
        if len(waiting_times) < 2:
            return 0.0
        
        std_wait = np.std(waiting_times)
        max_wait = np.max(waiting_times)
        
        # Normalize max wait to be comparable to std
        if self.config.normalize:
            max_wait = max_wait / 100.0  # Scale to similar magnitude
        
        r_fair = -(self.config.gamma * std_wait + 
                   self.config.delta * max_wait)
        
        return r_fair
    
    def _compute_penalty(self, waiting_times: np.ndarray) -> float:
        """
        Compute exponential penalty for starvation.
        
        R_penalty = -Σ exp(λ * (w_i - threshold)) for w_i > threshold
        
        This creates a strong incentive to serve vehicles waiting too long.
        """
        threshold = self.config.threshold
        lambda_ = self.config.lambda_
        
        # Find vehicles exceeding threshold
        starved_waits = waiting_times[waiting_times > threshold]
        
        if len(starved_waits) == 0:
            return 0.0
        
        # Exponential penalty grows rapidly with wait time
        penalties = np.exp(lambda_ * (starved_waits - threshold))
        total_penalty = -np.sum(penalties)
        
        return total_penalty
    
    def _update_stats(self, waiting_times: np.ndarray, 
                     queue_lengths: np.ndarray):
        """Update episode statistics for analysis"""
        self.episode_stats['waiting_times'].extend(waiting_times.tolist())
        self.episode_stats['queue_lengths'].extend(queue_lengths.tolist())
        self.episode_stats['starvation_count'] += np.sum(
            waiting_times > self.config.threshold
        )
        self.episode_stats['max_wait'] = max(
            self.episode_stats['max_wait'],
            np.max(waiting_times)
        )
    
    def _empty_info(self) -> Dict:
        """Return info dict for empty state"""
        return {
            'reward_total': 0.0,
            'reward_efficiency': 0.0,
            'reward_fairness': 0.0,
            'reward_penalty': 0.0,
            'avg_waiting_time': 0.0,
            'std_waiting_time': 0.0,
            'max_waiting_time': 0.0,
            'avg_queue_length': 0.0,
            'num_vehicles': 0,
            'starvation_count': 0
        }
    
    def get_episode_metrics(self) -> Dict:
        """
        Get aggregated metrics for the entire episode.
        Useful for evaluation and comparison.
        """
        if not self.episode_stats['waiting_times']:
            return {}
        
        waiting_times = np.array(self.episode_stats['waiting_times'])
        queue_lengths = np.array(self.episode_stats['queue_lengths'])
        
        # Fairness metrics
        gini_coef = self._compute_gini(waiting_times)
        jain_index = self._compute_jain_fairness(waiting_times)
        
        return {
            # Efficiency metrics
            'mean_waiting_time': np.mean(waiting_times),
            'median_waiting_time': np.median(waiting_times),
            'mean_queue_length': np.mean(queue_lengths),
            
            # Fairness metrics
            'std_waiting_time': np.std(waiting_times),
            'cv_waiting_time': np.std(waiting_times) / np.mean(waiting_times),
            'gini_coefficient': gini_coef,
            'jain_fairness_index': jain_index,
            
            # Extreme cases
            'max_waiting_time': self.episode_stats['max_wait'],
            'p95_waiting_time': np.percentile(waiting_times, 95),
            'p99_waiting_time': np.percentile(waiting_times, 99),
            'starvation_count': self.episode_stats['starvation_count'],
            'starvation_rate': (
                self.episode_stats['starvation_count'] / len(waiting_times)
            ),
            
            # Distribution
            'waiting_time_histogram': np.histogram(
                waiting_times, bins=10, range=(0, 300)
            )[0].tolist()
        }
    
    @staticmethod
    def _compute_gini(values: np.ndarray) -> float:
        """
        Compute Gini coefficient (inequality measure).
        0 = perfect equality, 1 = maximum inequality
        """
        if len(values) == 0:
            return 0.0
        
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        
        gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        return gini
    
    @staticmethod
    def _compute_jain_fairness(values: np.ndarray) -> float:
        """
        Compute Jain's fairness index.
        1 = perfect fairness, approaches 0 as unfairness increases
        """
        if len(values) == 0:
            return 1.0
        
        n = len(values)
        sum_values = np.sum(values)
        sum_squares = np.sum(values ** 2)
        
        if sum_squares == 0:
            return 1.0
        
        jain = (sum_values ** 2) / (n * sum_squares)
        return jain


class BaselineReward:
    """Standard efficiency-only reward for comparison"""
    
    def __init__(self):
        self.episode_waits = []
    
    def reset(self):
        self.episode_waits = []
    
    def compute_reward(self, state_info: Dict) -> Tuple[float, Dict]:
        """Simple efficiency-based reward"""
        waiting_times = np.array(state_info['waiting_times'])
        
        if len(waiting_times) == 0:
            return 0.0, {}
        
        # Standard reward: negative average waiting time
        reward = -np.mean(waiting_times)
        
        self.episode_waits.extend(waiting_times.tolist())
        
        info = {
            'reward_total': reward,
            'avg_waiting_time': np.mean(waiting_times),
            'max_waiting_time': np.max(waiting_times),
            'num_vehicles': len(waiting_times)
        }
        
        return reward, info
    
    def get_episode_metrics(self) -> Dict:
        if not self.episode_waits:
            return {}
        
        waits = np.array(self.episode_waits)
        return {
            'mean_waiting_time': np.mean(waits),
            'std_waiting_time': np.std(waits),
            'max_waiting_time': np.max(waits),
            'p95_waiting_time': np.percentile(waits, 95)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the reward function
    config = RewardConfig(
        w_efficiency=0.4,
        w_fairness=0.4,
        w_penalty=0.2,
        threshold=120.0
    )
    
    reward_fn = FairnessAwareReward(config)
    
    # Simulate state with varied waiting times (some vehicles starving)
    state_info = {
        'waiting_times': [10, 15, 20, 25, 30, 150, 180],  # Two starved
        'queue_lengths': [2, 3, 1, 2, 4, 0, 0],
        'vehicle_ids': ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
    }
    
    reward, info = reward_fn.compute_reward(state_info)
    
    print("Reward Components:")
    print(f"  Total: {reward:.3f}")
    print(f"  Efficiency: {info['reward_efficiency']:.3f}")
    print(f"  Fairness: {info['reward_fairness']:.3f}")
    print(f"  Penalty: {info['reward_penalty']:.3f}")
    print(f"\nMetrics:")
    print(f"  Avg wait: {info['avg_waiting_time']:.1f}s")
    print(f"  Std wait: {info['std_waiting_time']:.1f}s")
    print(f"  Max wait: {info['max_waiting_time']:.1f}s")
    print(f"  Starved: {info['starvation_count']}")
