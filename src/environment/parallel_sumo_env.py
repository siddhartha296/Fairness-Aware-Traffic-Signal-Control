"""
Parallel SUMO Environment Wrapper
src/environment/parallel_sumo_env.py

Uses Gymnasium VectorEnv to run multiple SUMO simulations in parallel.
"""

import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from typing import Callable, List, Optional
from pathlib import Path
import multiprocessing as mp

from .sumo_env import SumoTrafficEnv
from .reward import FairnessAwareReward, RewardConfig


def make_env(
    net_file: str,
    route_file: str,
    reward_config: RewardConfig,
    env_id: int,
    seed: int = None,
    **env_kwargs
) -> Callable:
    """
    Create a function that returns a SUMO environment.
    
    Each environment gets a unique seed and SUMO label to avoid conflicts.
    """
    def _init():
        # Create reward function for this environment
        reward_fn = FairnessAwareReward(reward_config)
        
        # Set unique seed for this environment
        env_seed = (seed + env_id) if seed is not None else None
        
        env = SumoTrafficEnv(
            net_file=net_file,
            route_file=route_file,
            reward_fn=reward_fn,
            use_gui=False,  # Never use GUI in parallel mode
            sumo_seed=env_seed,
            **env_kwargs
        )
        
        # Override the label to be unique per worker
        env.label = f"sumo_worker_{env_id}_{id(env)}"
        
        return env
    
    return _init


class ParallelSumoEnv:
    """
    Wrapper for parallel SUMO environments.
    
    Creates multiple SUMO instances running in parallel subprocesses.
    Collects experiences from all environments simultaneously.
    """
    
    def __init__(
        self,
        net_file: str,
        route_file: str,
        reward_config: RewardConfig,
        num_envs: int = 8,
        async_mode: bool = True,
        seed: int = None,
        **env_kwargs
    ):
        """
        Initialize parallel environments.
        
        Args:
            net_file: Path to SUMO network file
            route_file: Path to SUMO route file
            reward_config: Reward function configuration
            num_envs: Number of parallel environments
            async_mode: Use AsyncVectorEnv (faster) vs SyncVectorEnv
            seed: Base random seed
            **env_kwargs: Additional arguments for SumoTrafficEnv
        """
        self.num_envs = num_envs
        self.net_file = net_file
        self.route_file = route_file
        self.reward_config = reward_config
        
        print(f"\nCreating {num_envs} parallel SUMO environments...")
        print(f"  Mode: {'Async' if async_mode else 'Sync'}")
        
        # Create environment factories
        env_fns = [
            make_env(
                net_file=net_file,
                route_file=route_file,
                reward_config=reward_config,
                env_id=i,
                seed=seed,
                **env_kwargs
            )
            for i in range(num_envs)
        ]
        
        # Create vector environment
        if async_mode:
            self.vec_env = AsyncVectorEnv(env_fns)
        else:
            self.vec_env = SyncVectorEnv(env_fns)
        
        # Copy spaces from vector env
        self.observation_space = self.vec_env.single_observation_space
        self.action_space = self.vec_env.single_action_space
        
        print(f"✓ Parallel environments created")
        print(f"  Observation space: {self.observation_space}")
        print(f"  Action space: {self.action_space}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset all environments"""
        states, infos = self.vec_env.reset(seed=seed, options=options)
        return states, infos
    
    def step(self, actions):
        """
        Step all environments with given actions.
        
        Args:
            actions: Array of actions, one per environment
            
        Returns:
            states: Array of next states [num_envs, state_dim]
            rewards: Array of rewards [num_envs]
            terminateds: Array of terminated flags [num_envs]
            truncateds: Array of truncated flags [num_envs]
            infos: List of info dicts [num_envs]
        """
        return self.vec_env.step(actions)
    
    def close(self):
        """Close all environments"""
        self.vec_env.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close()
        except:
            pass


class ParallelReplayBuffer:
    """
    Replay buffer optimized for parallel experience collection.
    
    Stores transitions from multiple environments efficiently.
    """
    
    def __init__(self, capacity: int, num_envs: int):
        """
        Initialize buffer.
        
        Args:
            capacity: Total buffer capacity
            num_envs: Number of parallel environments
        """
        self.capacity = capacity
        self.num_envs = num_envs
        self.buffer = []
        self.position = 0
        
    def push_batch(self, states, actions, rewards, next_states, dones):
        """
        Push a batch of transitions from parallel environments.
        
        Args:
            states: [num_envs, state_dim]
            actions: [num_envs]
            rewards: [num_envs]
            next_states: [num_envs, state_dim]
            dones: [num_envs]
        """
        batch_size = len(states)
        
        for i in range(batch_size):
            experience = (
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                dones[i]
            )
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience
            
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample random batch of transitions"""
        import random
        import torch
        
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.tensor(
            np.vstack([e[0] for e in batch]), dtype=torch.float32
        )
        actions = torch.tensor(
            np.vstack([e[1] for e in batch]), dtype=torch.int64
        )
        rewards = torch.tensor(
            np.vstack([e[2] for e in batch]), dtype=torch.float32
        )
        next_states = torch.tensor(
            np.vstack([e[3] for e in batch]), dtype=torch.float32
        )
        dones = torch.tensor(
            np.vstack([e[4] for e in batch]), dtype=torch.float32
        )
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# Example usage
if __name__ == "__main__":
    # Test parallel environments
    network_path = Path("data/sumo_networks/single_intersection/medium")
    
    reward_config = RewardConfig()
    
    # Create parallel environments
    parallel_env = ParallelSumoEnv(
        net_file=str(network_path / "network.net.xml"),
        route_file=str(network_path / "routes.rou.xml"),
        reward_config=reward_config,
        num_envs=4,  # 4 parallel environments
        episode_length=360,  # Shorter for testing
    )
    
    print("\nTesting parallel step...")
    states, infos = parallel_env.reset()
    print(f"States shape: {states.shape}")  # Should be [4, state_dim]
    
    # Take random actions in all environments
    actions = np.array([
        parallel_env.action_space.sample() for _ in range(4)
    ])
    
    next_states, rewards, dones, truncated, infos = parallel_env.step(actions)
    
    print(f"Next states shape: {next_states.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Rewards: {rewards}")
    
    parallel_env.close()
    print("\n✓ Parallel environment test complete!")