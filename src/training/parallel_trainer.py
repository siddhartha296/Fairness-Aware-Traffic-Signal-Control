"""
Parallel Training Loop - Optimized for Multi-Core
src/training/parallel_trainer.py

Uses parallel SUMO environments to dramatically speed up training.
"""

import time
import numpy as np
from collections import deque
from tqdm import tqdm


class ParallelTrainer:
    """
    Trainer optimized for parallel environments.
    
    Collects experiences from multiple environments simultaneously,
    then performs batch updates on GPU.
    """
    
    def __init__(self, agent, parallel_env, logger, tb_logger, 
                 checkpoint_mgr, config: dict):
        """
        Initialize parallel trainer.
        
        Args:
            agent: RL agent
            parallel_env: ParallelSumoEnv instance
            logger: Logger
            tb_logger: TensorBoard logger
            checkpoint_mgr: Checkpoint manager
            config: Training configuration
        """
        self.agent = agent
        self.parallel_env = parallel_env
        self.num_envs = parallel_env.num_envs
        self.logger = logger
        self.tb_logger = tb_logger
        self.checkpoint_mgr = checkpoint_mgr
        self.config = config
        
        # Training parameters
        self.target_update_freq = config.get('target_update_freq', 1000)
        self.eval_freq = config.get('eval_freq', 100)
        self.checkpoint_freq = config.get('checkpoint_freq', 500)
        self.log_interval = config.get('log_interval', 10)
        
        # Stats tracking
        self.total_steps = 0
        self.episode_count = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Per-environment tracking
        self.env_episode_rewards = [0.0] * self.num_envs
        self.env_episode_lengths = [0] * self.num_envs
        
        print(f"\n✓ Parallel trainer initialized for {self.num_envs} environments")
    
    def train(self, num_episodes: int, start_episode: int = 0, 
              batch_size: int = 128, update_freq: int = 4):
        """
        Main parallel training loop.
        
        Args:
            num_episodes: Total episodes to train (across all envs)
            start_episode: Starting episode number
            batch_size: Batch size for agent updates
            update_freq: Perform agent update every N steps
        """
        
        self.agent.batch_size = batch_size
        self.episode_count = start_episode
        
        self.logger.info(
            f"Starting parallel training from episode {start_episode + 1} "
            f"for {num_episodes} total episodes."
        )
        self.logger.info(
            f"Using {self.num_envs} parallel environments "
            f"(~{num_episodes // self.num_envs} episodes per env)"
        )
        
        # Reset all environments
        states, infos = self.parallel_env.reset()
        
        # Training loop with progress bar
        pbar = tqdm(
            total=num_episodes,
            initial=start_episode,
            desc="Training",
            unit="ep"
        )
        
        start_time = time.time()
        step_count = 0
        
        try:
            while self.episode_count < num_episodes:
                # Select actions for all environments
                actions = np.array([
                    self.agent.select_action(states[i], eval_mode=False)
                    for i in range(self.num_envs)
                ])
                
                # Step all environments
                next_states, rewards, dones, truncateds, infos = \
                    self.parallel_env.step(actions)
                
                # Store transitions in replay buffer
                # Push entire batch at once for efficiency
                for i in range(self.num_envs):
                    self.agent.store_transition(
                        states[i],
                        actions[i],
                        rewards[i],
                        next_states[i],
                        dones[i] or truncateds[i]
                    )
                    
                    # Track episode stats
                    self.env_episode_rewards[i] += rewards[i]
                    self.env_episode_lengths[i] += 1
                    
                    # Handle episode completion
                    if dones[i] or truncateds[i]:
                        self.episode_rewards.append(self.env_episode_rewards[i])
                        self.episode_lengths.append(self.env_episode_lengths[i])
                        
                        self.episode_count += 1
                        pbar.update(1)
                        
                        # Log to TensorBoard
                        self.tb_logger.log_scalar(
                            'Reward/Episode',
                            self.env_episode_rewards[i],
                            self.episode_count
                        )
                        
                        # Reset episode stats for this env
                        self.env_episode_rewards[i] = 0.0
                        self.env_episode_lengths[i] = 0
                        
                        # Check if we've reached target episodes
                        if self.episode_count >= num_episodes:
                            break
                
                # Train agent
                if step_count % update_freq == 0:
                    loss = self.agent.train_step()
                
                # Update target network
                if self.total_steps % self.target_update_freq == 0:
                    self.agent.update_target_network()
                
                # Update state
                states = next_states
                step_count += 1
                self.total_steps += self.num_envs  # Each step advances all envs
                
                # Periodic logging
                if self.episode_count % self.log_interval == 0 and \
                   self.episode_count > start_episode:
                    self._log_progress()
                
                # Checkpointing
                if self.episode_count % self.checkpoint_freq == 0 and \
                   self.episode_count > start_episode:
                    self._save_checkpoint()
            
            pbar.close()
            
            # Update epsilon at end
            self.agent.update_epsilon()
            
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Training complete in {elapsed_time:.1f}s "
                f"({elapsed_time / 60:.1f} min)"
            )
            self.logger.info(
                f"Episodes per second: "
                f"{(num_episodes - start_episode) / elapsed_time:.2f}"
            )
            
        except KeyboardInterrupt:
            pbar.close()
            self.logger.info("Training interrupted by user")
            self._save_checkpoint(name="interrupted_model.pth")
    
    def _log_progress(self):
        """Log training progress"""
        if len(self.episode_rewards) == 0:
            return
        
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        
        self.tb_logger.log_scalar(
            'Reward/Average_100',
            avg_reward,
            self.episode_count
        )
        self.tb_logger.log_scalar(
            'Params/Epsilon',
            self.agent.epsilon,
            self.episode_count
        )
        
        self.logger.info(
            f"Ep: {self.episode_count}/{self.config.get('episodes', 'N/A')} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Avg Length: {avg_length:.0f} | "
            f"Epsilon: {self.agent.epsilon:.3f} | "
            f"Buffer: {len(self.agent.memory)}"
        )
    
    def _save_checkpoint(self, name: str = None):
        """Save training checkpoint"""
        if name is None:
            name = f"checkpoint_ep{self.episode_count}.pth"
        
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        
        self.checkpoint_mgr.save(
            self.agent,
            episode=self.episode_count,
            metrics={'avg_reward': avg_reward}
        )
    
    def print_summary(self):
        """Print training summary"""
        if len(self.episode_rewards) > 0:
            self.logger.info("Training complete.")
            self.logger.info(
                f"Final Avg Reward (last 100): "
                f"{np.mean(self.episode_rewards):.2f}"
            )
            self.logger.info(
                f"Total steps: {self.total_steps:,}"
            )


# Modified DQN Agent for parallel training
def make_parallel_agent(state_dim, action_dim, config, device):
    """
    Create DQN agent optimized for parallel training.
    
    Uses a larger replay buffer since we're collecting experiences faster.
    """
    from src.models.dqn import DQNAgent
    
    model_config = config['model']
    
    # Increase buffer size for parallel collection
    buffer_size = model_config.get('replay_buffer_size', 100000)
    parallel_buffer_size = buffer_size * 2  # Larger buffer for parallel
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=model_config['architecture']['hidden_layers'],
        learning_rate=model_config['learning_rate'],
        gamma=model_config['gamma'],
        epsilon_start=model_config['epsilon_start'],
        epsilon_end=model_config['epsilon_end'],
        epsilon_decay=model_config['epsilon_decay'],
        buffer_size=parallel_buffer_size,
        device=device
    )
    
    return agent