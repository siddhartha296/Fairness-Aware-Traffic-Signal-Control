"""
Training Loop
src/training/trainer.py
"""

import time
import numpy as np
from collections import deque

class Trainer:
    """Trainer class to handle the training loop"""
    
    def __init__(self, agent, env, logger, tb_logger, checkpoint_mgr, config: dict):
        self.agent = agent
        self.env = env
        self.logger = logger
        self.tb_logger = tb_logger
        self.checkpoint_mgr = checkpoint_mgr
        self.config = config
        
        # Training parameters
        self.target_update_freq = config.get('target_update_freq', 1000)
        self.eval_freq = config.get('eval_freq', 100)
        self.num_eval_episodes = config.get('num_eval_episodes', 10)
        self.checkpoint_freq = config.get('checkpoint_freq', 500)
        
        # Stats
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_fairness_metrics = deque(maxlen=100) # e.g., std_wait
        
    def train(self, num_episodes: int, start_episode: int = 0, batch_size: int = 128):
        """Main training loop"""
        
        self.agent.batch_size = batch_size
        
        self.logger.info(f"Starting training from episode {start_episode + 1} for {num_episodes} episodes.")
        
        for episode in range(start_episode, num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            truncated = False
            
            start_time = time.time()
            
            while not (done or truncated):
                # Select action
                action = self.agent.select_action(state, eval_mode=False)
                
                # Step environment
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done or truncated)
                
                # Train
                loss = self.agent.train_step()
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1
                
                # Update target network
                if self.total_steps % self.target_update_freq == 0:
                    self.agent.update_target_network()
                    
            # End of episode
            self.agent.update_epsilon()
            self.episode_rewards.append(episode_reward)
            
            # Log metrics
            metrics = self.env.get_metrics()
            std_wait = metrics.get('std_waiting_time', 0)
            self.episode_fairness_metrics.append(std_wait)
            
            avg_reward = np.mean(self.episode_rewards)
            avg_std_wait = np.mean(self.episode_fairness_metrics)
            
            self.tb_logger.log_scalar('Reward/Episode', episode_reward, episode)
            self.tb_logger.log_scalar('Reward/Average_100', avg_reward, episode)
            self.tb_logger.log_scalar('Metrics/Avg_Std_Wait', avg_std_wait, episode)
            self.tb_logger.log_scalar('Params/Epsilon', self.agent.epsilon, episode)
            
            if (episode + 1) % self.config.get('log_interval', 10) == 0:
                self.logger.info(
                    f"Ep: {episode + 1}/{num_episodes} | "
                    f"Steps: {episode_steps} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Std Wait: {avg_std_wait:.2f} | "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )
                
            # Evaluate
            if (episode + 1) % self.eval_freq == 0:
                self.evaluate(episode)
                
            # Save checkpoint
            if (episode + 1) % self.checkpoint_freq == 0:
                self.checkpoint_mgr.save(
                    self.agent,
                    episode=episode + 1,
                    metrics={'avg_reward': avg_reward}
                )
        
    def evaluate(self, current_episode: int):
        """Run evaluation episodes"""
        self.logger.info(f"Running evaluation at episode {current_episode + 1}...")
        eval_rewards = []
        eval_metrics = []
        
        for _ in range(self.num_eval_episodes):
            state, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                action = self.agent.select_action(state, eval_mode=True)
                state, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                
            eval_rewards.append(episode_reward)
            eval_metrics.append(self.env.get_metrics())
            
        mean_eval_reward = np.mean(eval_rewards)
        
        # Log aggregated eval metrics
        mean_eval_std = np.mean([m.get('std_waiting_time', 0) for m in eval_metrics])
        mean_eval_max_wait = np.mean([m.get('max_waiting_time', 0) for m in eval_metrics])
        
        self.logger.info(
            f"Evaluation: Avg Reward: {mean_eval_reward:.2f} | "
            f"Avg Std Wait: {mean_eval_std:.2f} | "
            f"Avg Max Wait: {mean_eval_max_wait:.2f}"
        )
        
        self.tb_logger.log_scalar('Evaluation/Avg_Reward', mean_eval_reward, current_episode)
        self.tb_logger.log_scalar('Evaluation/Avg_Std_Wait', mean_eval_std, current_episode)
        self.tb_logger.log_scalar('Evaluation/Avg_Max_Wait', mean_eval_max_wait, current_episode)
        
    def print_summary(self):
        self.logger.info("Training complete.")
        self.logger.info(f"Final Avg Reward (last 100): {np.mean(self.episode_rewards):.2f}")
