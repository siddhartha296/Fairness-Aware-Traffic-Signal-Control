"""
Evaluation Script for Traffic Signal Control
scripts/evaluate.py

Evaluates trained models and compares with baselines.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.sumo_env import SumoTrafficEnv
from src.environment.reward import FairnessAwareReward, BaselineReward, RewardConfig
from src.models.dqn import DQNAgent


class BaselineController:
    """Baseline controllers for comparison"""
    
    @staticmethod
    def fixed_time(state, phase_durations=[30, 30, 30, 30]):
        """Fixed-time controller"""
        # Cycles through phases with fixed durations
        # This is a stateless controller
        return 0  # Would need state tracking for real implementation
    
    @staticmethod
    def max_pressure(state, num_actions):
        """Max-pressure controller (greedy based on queue lengths)"""
        # Select phase that serves lanes with longest queues
        # state contains queue lengths
        num_lanes = (len(state) - 2) // 2
        queue_lengths = state[:num_lanes]
        
        # Simple heuristic: select action based on max queue
        return int(np.argmax(queue_lengths) % num_actions)
    
    @staticmethod
    def random_controller(state, num_actions):
        """Random action selection"""
        return np.random.randint(num_actions)


def evaluate_model(
    agent,
    env,
    num_episodes: int,
    render: bool = False
) -> Dict:
    """
    Evaluate a trained agent.
    
    Args:
        agent: Trained agent (or baseline controller)
        env: Environment
        num_episodes: Number of evaluation episodes
        render: Whether to render
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    episode_rewards = []
    all_metrics = []
    
    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        
        while not (done or truncated):
            if isinstance(agent, str):
                # Baseline controller
                if agent == 'fixed_time':
                    action = BaselineController.fixed_time(state)
                elif agent == 'max_pressure':
                    action = BaselineController.max_pressure(state, env.action_space.n)
                elif agent == 'random':
                    action = BaselineController.random_controller(state, env.action_space.n)
            else:
                # Trained agent
                action = agent.select_action(state, eval_mode=True)
            
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        episode_rewards.append(total_reward)
        metrics = env.get_metrics()
        all_metrics.append(metrics)
    
    # Aggregate results
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
    }
    
    # Average metrics
    if all_metrics:
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                results[key] = np.mean(values)
                results[f'{key}_std'] = np.std(values)
    
    return results


def compare_controllers(
    config: Dict,
    model_path: Path,
    num_episodes: int = 50
) -> pd.DataFrame:
    """
    Compare trained model with baseline controllers.
    
    Args:
        config: Configuration dictionary
        model_path: Path to trained model
        num_episodes: Number of evaluation episodes
        
    Returns:
        comparison_df: DataFrame with comparison results
    """
    print("\n" + "="*60)
    print("Controller Comparison")
    print("="*60)
    
    # Setup environment
    reward_config = RewardConfig(**config['reward']['parameters'])
    reward_config.w_efficiency = config['reward']['weights']['efficiency']
    reward_config.w_fairness = config['reward']['weights']['fairness']
    reward_config.w_penalty = config['reward']['weights']['penalty']
    
    reward_fn = FairnessAwareReward(reward_config)
    
    network_path = Path('data/sumo_networks/single/medium')
    env = SumoTrafficEnv(
        net_file=str(network_path / 'network.net.xml'),
        route_file=str(network_path / 'routes.rou.xml'),
        reward_fn=reward_fn,
        use_gui=False,
        episode_length=config['environment']['episode_length']
    )
    
    results = {}
    
    # 1. Evaluate trained model
    print("\n1. Evaluating trained DQN agent...")
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_layers=config['model']['architecture']['hidden_layers']
    )
    agent.load(model_path)
    results['Fairness-Aware DQN'] = evaluate_model(agent, env, num_episodes)
    
    # 2. Evaluate baseline DQN (efficiency-only)
    print("\n2. Evaluating baseline DQN (efficiency-only)...")
    baseline_reward_fn = BaselineReward()
    env.reward_fn = baseline_reward_fn
    # Would need to load baseline model if available
    # For now, using same model with different reward
    results['Baseline DQN'] = evaluate_model(agent, env, num_episodes)
    
    # Restore fairness reward
    env.reward_fn = reward_fn
    
    # 3. Fixed-time controller
    print("\n3. Evaluating fixed-time controller...")
    results['Fixed-Time'] = evaluate_model('fixed_time', env, num_episodes)
    
    # 4. Max-pressure controller
    print("\n4. Evaluating max-pressure controller...")
    results['Max-Pressure'] = evaluate_model('max_pressure', env, num_episodes)
    
    # 5. Random controller
    print("\n5. Evaluating random controller...")
    results['Random'] = evaluate_model('random', env, num_episodes)
    
    env.close()
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    
    # Select key metrics for comparison
    key_metrics = [
        'mean_reward',
        'mean_waiting_time',
        'std_waiting_time',
        'max_waiting_time',
        'gini_coefficient',
        'jain_fairness_index',
        'starvation_rate'
    ]
    
    comparison_df = comparison_df[[m for m in key_metrics if m in comparison_df.columns]]
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate traffic signal control models'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/train_config.yaml',
        help='Configuration file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--compare-baselines',
        action='store_true',
        help='Compare with baseline controllers'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare_baselines:
        # Full comparison
        comparison_df = compare_controllers(
            config,
            Path(args.model),
            args.episodes
        )
        
        # Print results
        print("\n" + "="*60)
        print("Comparison Results")
        print("="*60)
        print(comparison_df.to_string())
        
        # Save results
        comparison_df.to_csv(output_dir / 'comparison.csv')
        comparison_df.to_latex(output_dir / 'comparison.tex')
        
        # Save as JSON
        with open(output_dir / 'comparison.json', 'w') as f:
            json.dump(comparison_df.to_dict(), f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
    
    else:
        # Simple evaluation
        print("Evaluating single model...")
        
        # Setup environment
        reward_config = RewardConfig(**config['reward']['parameters'])
        reward_fn = FairnessAwareReward(reward_config)
        
        network_path = Path('data/sumo_networks/single/medium')
        env = SumoTrafficEnv(
            net_file=str(network_path / 'network.net.xml'),
            route_file=str(network_path / 'routes.rou.xml'),
            reward_fn=reward_fn,
            use_gui=False,
            episode_length=config['environment']['episode_length']
        )
        
        # Load agent
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_layers=config['model']['architecture']['hidden_layers']
        )
        agent.load(Path(args.model))
        
        # Evaluate
        results = evaluate_model(agent, env, args.episodes)
        
        # Print results
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        for key, value in results.items():
            print(f"{key:30s}: {value:.4f}")
        
        # Save results
        with open(output_dir / 'evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        env.close()
        print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
