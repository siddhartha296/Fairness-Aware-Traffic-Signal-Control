"""
Inference Script
scripts/inference.py

Run a trained model, optionally with GUI.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.sumo_env import SumoTrafficEnv
from src.environment.reward import FairnessAwareReward, RewardConfig
from src.models.dqn import DQNAgent

def main():
    parser = argparse.ArgumentParser(
        description='Run inference with a trained model'
    )
    parser.add_argument(
        '--model', type=str, required=True, help='Path to trained model (.pth)'
    )
    parser.add_argument(
        '--config', type=str, default='config/train_config.yaml', help='Path to training config'
    )
    parser.add_argument(
        '--network-dir', type=str, required=True, help='Path to SUMO network directory (e.g., data/sumo_networks/single/medium)'
    )
    parser.add_argument(
        '--episodes', type=int, default=10, help='Number of episodes to run'
    )
    parser.add_argument(
        '--visualize', action='store_true', help='Run with SUMO GUI'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup environment
    network_path = Path(args.network_dir)
    if not (network_path / "network.net.xml").exists():
        print(f"Error: Network file not found in {network_path}")
        sys.exit(1)
        
    reward_config = RewardConfig(**config['reward']['parameters'])
    reward_fn = FairnessAwareReward(reward_config)
    
    env = SumoTrafficEnv(
        net_file=str(network_path / 'network.net.xml'),
        route_file=str(network_path / 'routes.rou.xml'),
        reward_fn=reward_fn,
        use_gui=args.visualize,
        episode_length=config['environment']['episode_length']
    )
    
    # Create agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_layers=config['model']['architecture']['hidden_layers'],
        device=device
    )
    
    # Load trained model
    agent.load(Path(args.model))
    print(f"Model loaded from {args.model}")
    
    # Run inference
    print(f"Running {args.episodes} episodes...")
    all_metrics = []
    
    for ep in range(args.episodes):
        state, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action = agent.select_action(state, eval_mode=True)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
        metrics = env.get_metrics()
        all_metrics.append(metrics)
        print(
            f"Episode {ep + 1}: Reward={episode_reward:.2f}, "
            f"AvgWait={metrics.get('mean_waiting_time', 0):.2f}, "
            f"MaxWait={metrics.get('max_waiting_time', 0):.2f}, "
            f"StarveRate={metrics.get('starvation_rate', 0):.3f}"
        )
        
    env.close()
    
    # Print average results
    print("\n" + "="*60)
    print("Inference Complete: Average Metrics")
    print("="*60)
    
    avg_results = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
        if values:
            avg_results[key] = np.mean(values)
            
    for key, value in avg_results.items():
        print(f"{key:25s}: {value:.3f}")

if __name__ == '__main__':
    main()
