"""
Checkpoint Manager
src/utils/checkpoint.py
"""

import torch
from pathlib import Path
from typing import Dict, Optional

class CheckpointManager:
    """Manages saving and pruning model checkpoints"""
    
    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = [] # List of (metric, path)
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, agent, episode: int, metrics: Dict):
        """Save a new checkpoint"""
        
        filename = f"checkpoint_ep{episode}.pth"
        filepath = self.checkpoint_dir / filename
        
        print(f"\nSaving checkpoint to {filepath}...")
        
        checkpoint_data = {
            'agent_state_dict': agent.policy_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'episode': episode,
            'metrics': metrics
        }
        
        torch.save(checkpoint_data, filepath)
        
        # Add to list and prune
        # Use a key metric for pruning, e.g., avg_reward
        metric_val = metrics.get('avg_reward', -float('inf'))
        self.checkpoints.append((metric_val, filepath))
        
        self._prune()
        
    def _prune(self):
        """Remove worst checkpoints if exceeding max"""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by metric (ascending)
            self.checkpoints.sort(key=lambda x: x[0])
            
            # Remove worst
            worst_metric, worst_path = self.checkpoints.pop(0)
            
            if worst_path.exists():
                worst_path.unlink()
                print(f"Removed old checkpoint: {worst_path.name} (Metric: {worst_metric:.2f})")
