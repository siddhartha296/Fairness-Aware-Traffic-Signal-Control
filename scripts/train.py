"""
Main Training Script for Fairness-Aware Traffic Signal Control
scripts/train.py

Run on GPU server:
python scripts/train.py --config config/train_config.yaml --gpu 0,1
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.sumo_env import SumoTrafficEnv
from src.environment.reward import FairnessAwareReward, RewardConfig
from src.models.dqn import DQNAgent
from src.training.trainer import Trainer
from src.utils.logger import setup_logger, TensorboardLogger
from src.utils.checkpoint import CheckpointManager

# For reproducibility
def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Fairness-Aware Traffic Signal Controller'
    )
    
    # Config
    parser.add_argument(
        '--config',
        type=str,
        default='config/train_config.yaml',
        help='Path to config file'
    )
    
    # Environment
    parser.add_argument(
        '--network',
        type=str,
        default='single_intersection',
        choices=['single_intersection', 'grid_4x4', 'arterial'],
        help='Traffic network to use'
    )
    
    # Training
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of training episodes (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='DQN',
        choices=['DQN', 'PPO'],
        help='RL algorithm to use'
    )
    
    # Hardware
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU IDs to use (comma-separated, e.g., "0,1")'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of data loading workers'
    )
    
    # Checkpointing
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=None,
        help='Save checkpoint every N episodes'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    # Experiment
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name (for logging)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Debugging
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (fewer episodes, verbose logging)'
    )
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run SUMO without GUI (headless mode)'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(gpu_ids: str) -> torch.device:
    """Setup GPU device(s)"""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    
    # Parse GPU IDs
    gpu_list = [int(i) for i in gpu_ids.split(',')]
    
    # Set visible devices
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    
    device = torch.device(f'cuda:0')
    print(f"Using GPU(s): {gpu_ids}")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def create_experiment_dir(args, config: Dict) -> Path:
    """Create directory for experiment results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.name:
        exp_name = f"{args.name}_{timestamp}"
    else:
        exp_name = f"{args.algorithm}_{args.network}_{timestamp}"
    
    exp_dir = Path('results') / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'visualizations').mkdir(exist_ok=True)
    
    # Save config
    config_copy = config.copy()
    config_copy['args'] = vars(args)
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config_copy, f, indent=2)
    
    print(f"\nExperiment directory: {exp_dir}")
    
    return exp_dir


def create_environment(args, config: Dict) -> SumoTrafficEnv:
    """Create SUMO traffic environment"""
    env_config = config['environment']
    
    # Get network path
    network_path = Path('data/sumo_networks') / args.network
    
    if not network_path.exists():
        raise FileNotFoundError(
            f"Network not found: {network_path}\n"
            f"Run: python scripts/generate_traffic.py --network {args.network}"
        )
    
    # Create reward function
    reward_config = RewardConfig(**config['reward']['parameters'])
    reward_config.w_efficiency = config['reward']['weights']['efficiency']
    reward_config.w_fairness = config['reward']['weights']['fairness']
    reward_config.w_penalty = config['reward']['weights']['penalty']
    
    reward_fn = FairnessAwareReward(reward_config)
    
    # Create environment
    env = SumoTrafficEnv(
        net_file=str(network_path / 'network.net.xml'),
        route_file=str(network_path / 'routes.rou.xml'),
        reward_fn=reward_fn,
        use_gui=not args.no_gui and not args.debug,
        episode_length=env_config['episode_length'],
        step_size=env_config['step_size'],
        yellow_time=env_config['yellow_time'],
        min_green=env_config['min_green'],
        max_green=env_config['max_green']
    )
    
    print(f"\nEnvironment created:")
    print(f"  Network: {args.network}")
    print(f"  State dim: {env.observation_space.shape}")
    print(f"  Action dim: {env.action_space.n}")
    print(f"  Episode length: {env_config['episode_length']}s")
    
    return env


def create_agent(args, config: Dict, env, device: torch.device):
    """Create RL agent"""
    model_config = config['model']
    
    if args.algorithm == 'DQN':
        from src.models.dqn import DQNAgent
        
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_layers=model_config['architecture']['hidden_layers'],
            learning_rate=model_config['learning_rate'],
            gamma=model_config['gamma'],
            epsilon_start=model_config['epsilon_start'],
            epsilon_end=model_config['epsilon_end'],
            epsilon_decay=model_config['epsilon_decay'],
            device=device
        )
    
    elif args.algorithm == 'PPO':
        from src.models.ppo import PPOAgent
        
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_layers=model_config['architecture']['hidden_layers'],
            learning_rate=model_config['learning_rate'],
            gamma=model_config['gamma'],
            device=device
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    print(f"\nAgent created:")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Total parameters: {agent.count_parameters():,}")
    
    return agent


def train(args, config: Dict):
    """Main training function"""
    
    # Setup
    set_seed(args.seed)
    device = setup_device(args.gpu)
    exp_dir = create_experiment_dir(args, config)
    
    # Logging
    logger = setup_logger(exp_dir / 'logs' / 'training.log')
    tb_logger = TensorboardLogger(exp_dir / 'logs' / 'tensorboard')
    
    # Create environment and agent
    env = create_environment(args, config)
    agent = create_agent(args, config, env, device)
    
    # Checkpoint manager
    checkpoint_mgr = CheckpointManager(
        exp_dir / 'checkpoints',
        max_checkpoints=5
    )
    
    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        start_episode = checkpoint['episode']
        logger.info(f"Resumed from episode {start_episode}")
    
    # Training config
    training_config = config['training']
    num_episodes = args.episodes or training_config['episodes']
    batch_size = args.batch_size or training_config['batch_size']
    
    if args.debug:
        num_episodes = 10
        print("\n[DEBUG MODE] Running only 10 episodes")
    
    # Create trainer
    trainer = Trainer(
        agent=agent,
        env=env,
        logger=logger,
        tb_logger=tb_logger,
        checkpoint_mgr=checkpoint_mgr,
        config=training_config
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    try:
        trainer.train(
            num_episodes=num_episodes,
            start_episode=start_episode,
            batch_size=batch_size
        )
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        # Save final model
        final_path = exp_dir / 'checkpoints' / 'final_model.pth'
        agent.save(final_path)
        print(f"Final model saved to: {final_path}")
        
        # Print final statistics
        trainer.print_summary()
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        
        # Save interrupted model
        interrupt_path = exp_dir / 'checkpoints' / 'interrupted_model.pth'
        agent.save(interrupt_path)
        print(f"Model saved to: {interrupt_path}")
    
    finally:
        # Cleanup
        env.close()
        tb_logger.close()
        
        print(f"\nResults saved to: {exp_dir}")


def main():
    """Main entry point"""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.episodes:
        config['training']['episodes'] = args.episodes
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.checkpoint_freq:
        config['training']['checkpoint_freq'] = args.checkpoint_freq
    
    # Print configuration
    print("\n" + "="*60)
    print("Fairness-Aware Traffic Signal Control")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Config file: {args.config}")
    print(f"  Network: {args.network}")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Episodes: {config['training']['episodes']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Seed: {args.seed}")
    print(f"  GPU: {args.gpu}")
    
    # Start training
    train(args, config)


if __name__ == '__main__':
    main()
