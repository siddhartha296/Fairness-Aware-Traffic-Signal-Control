"""
Parallel Training Script - FAST VERSION
scripts/train_parallel.py

Uses multiple SUMO environments in parallel for fast training.

Usage:
    python scripts/train_parallel.py \
        --config config/train_config.yaml \
        --network single_intersection \
        --algorithm DQN \
        --episodes 5000 \
        --num-workers 8 \
        --gpu 0

Speed improvement: ~8x faster with 8 workers!
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.parallel_sumo_env import ParallelSumoEnv
from src.environment.reward import RewardConfig
from src.models.dqn import DQNAgent
from src.training.parallel_trainer import ParallelTrainer
from src.utils.logger import setup_logger, TensorboardLogger
from src.utils.checkpoint import CheckpointManager


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train with Parallel SUMO Environments (FAST!)'
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
        help='Traffic network to use'
    )
    parser.add_argument(
        '--demand',
        type=str,
        default='medium',
        choices=['low', 'medium', 'high', 'rush_hour'],
        help='Traffic demand level'
    )
    
    # Parallel settings
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of parallel SUMO environments (default: 8)'
    )
    parser.add_argument(
        '--update-freq',
        type=int,
        default=4,
        help='Agent update frequency in steps (default: 4)'
    )
    
    # Training
    parser.add_argument(
        '--episodes',
        type=int,
        default=5000,
        help='Total number of training episodes'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size (larger for parallel training)'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='DQN',
        choices=['DQN'],
        help='RL algorithm'
    )
    
    # Hardware
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU ID to use'
    )
    
    # Checkpointing
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=250,
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
        default='parallel_training',
        help='Experiment name'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def setup_device(gpu_id: str) -> torch.device:
    """Setup GPU device"""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device('cuda:0')
    
    print(f"\nGPU Setup:")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_experiment_dir(args, config: dict) -> Path:
    """Create directory for experiment results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.name}_{timestamp}"
    
    exp_dir = Path('results') / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    
    # Save config
    config_copy = config.copy()
    config_copy['args'] = vars(args)
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config_copy, f, indent=2)
    
    print(f"\nExperiment directory: {exp_dir}")
    return exp_dir


def main():
    """Main entry point"""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with args
    if args.episodes:
        config['training']['episodes'] = args.episodes
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.checkpoint_freq:
        config['training']['checkpoint_freq'] = args.checkpoint_freq
    
    # Print configuration
    print("\n" + "="*70)
    print("PARALLEL TRAINING - FAIRNESS-AWARE TRAFFIC SIGNAL CONTROL")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Network: {args.network}")
    print(f"  Demand: {args.demand}")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Parallel workers: {args.num_workers}")
    print(f"  Update frequency: every {args.update_freq} steps")
    print(f"  GPU: {args.gpu}")
    
    # Estimate speedup
    print(f"\n⚡ Expected speedup: ~{args.num_workers}x faster than serial!")
    
    # Setup
    set_seed(args.seed)
    device = setup_device(args.gpu)
    exp_dir = create_experiment_dir(args, config)
    
    # Logging
    logger = setup_logger(exp_dir / 'logs' / 'training.log')
    tb_logger = TensorboardLogger(exp_dir / 'logs' / 'tensorboard')
    
    # Get network path
    network_path = Path('data/sumo_networks') / args.network / args.demand
    
    if not network_path.exists():
        print(f"\n❌ Network not found: {network_path}")
        print(f"Generate it with:")
        print(f"  python scripts/generate_traffic.py --network {args.network}")
        sys.exit(1)
    
    net_file = str(network_path / 'network.net.xml')
    route_file = str(network_path / 'routes.rou.xml')
    
    print(f"\nTraffic scenario:")
    print(f"  Network: {net_file}")
    print(f"  Routes: {route_file}")
    
    # Create reward config
    reward_config = RewardConfig(**config['reward']['parameters'])
    reward_config.w_efficiency = config['reward']['weights']['efficiency']
    reward_config.w_fairness = config['reward']['weights']['fairness']
    reward_config.w_penalty = config['reward']['weights']['penalty']
    
    # Create parallel environments
    env_kwargs = {
        'episode_length': config['environment']['episode_length'],
        'step_size': config['environment']['step_size'],
        'yellow_time': config['environment']['yellow_time'],
        'min_green': config['environment']['min_green'],
        'max_green': config['environment']['max_green'],
    }
    
    parallel_env = ParallelSumoEnv(
        net_file=net_file,
        route_file=route_file,
        reward_config=reward_config,
        num_envs=args.num_workers,
        async_mode=True,  # Async is faster
        seed=args.seed,
        **env_kwargs
    )
    
    # Create agent
    state_dim = parallel_env.observation_space.shape[0]
    action_dim = parallel_env.action_space.n
    
    print(f"\nAgent configuration:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=config['model']['architecture']['hidden_layers'],
        learning_rate=config['model']['learning_rate'],
        gamma=config['model']['gamma'],
        epsilon_start=config['model']['epsilon_start'],
        epsilon_end=config['model']['epsilon_end'],
        epsilon_decay=config['model']['epsilon_decay'],
        buffer_size=config['model']['replay_buffer_size'] * 2,  # Larger buffer
        device=device
    )
    
    print(f"  Total parameters: {agent.count_parameters():,}")
    
    # Checkpoint manager
    checkpoint_mgr = CheckpointManager(
        exp_dir / 'checkpoints',
        max_checkpoints=5
    )
    
    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume)
        agent.policy_net.load_state_dict(checkpoint['agent_state_dict'])
        agent.target_net.load_state_dict(checkpoint['agent_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        logger.info(f"Resumed from episode {start_episode}")
    
    # Create parallel trainer
    trainer = ParallelTrainer(
        agent=agent,
        parallel_env=parallel_env,
        logger=logger,
        tb_logger=tb_logger,
        checkpoint_mgr=checkpoint_mgr,
        config=config['training']
    )
    
    # Start training
    print("\n" + "="*70)
    print("STARTING PARALLEL TRAINING")
    print("="*70)
    print(f"\nMonitor with TensorBoard:")
    print(f"  tensorboard --logdir {exp_dir / 'logs' / 'tensorboard'} --port 6006")
    print()
    
    try:
        trainer.train(
            num_episodes=args.episodes,
            start_episode=start_episode,
            batch_size=args.batch_size,
            update_freq=args.update_freq
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        
        # Save final model
        final_path = exp_dir / 'checkpoints' / 'final_model.pth'
        agent.save(final_path)
        print(f"\n✓ Final model saved: {final_path}")
        
        trainer.print_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        interrupt_path = exp_dir / 'checkpoints' / 'interrupted_model.pth'
        agent.save(interrupt_path)
        print(f"✓ Model saved: {interrupt_path}")
    
    finally:
        parallel_env.close()
        tb_logger.close()
        print(f"\n✓ Results saved to: {exp_dir}")


if __name__ == '__main__':
    main()