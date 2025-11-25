"""
SUMO Traffic Environment Wrapper
src/environment/sumo_env.py
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import traci
import sumolib

from .reward import FairnessAwareReward, RewardConfig


class SumoTrafficEnv(gym.Env):
    """
    Gymnasium environment for traffic signal control using SUMO.
    
    State: Queue lengths, waiting times, phase info for each intersection
    Action: Traffic light phase selection
    Reward: Fairness-aware reward combining efficiency and equity
    """
    
    def __init__(
        self,
        net_file: str,
        route_file: str,
        reward_fn: FairnessAwareReward,
        use_gui: bool = False,
        episode_length: int = 3600,
        step_size: int = 5,
        yellow_time: int = 3,
        min_green: int = 10,
        max_green: int = 60,
        sumo_seed: int = None,
        sumo_warnings: bool = False
    ):
        """
        Initialize SUMO environment.
        
        Args:
            net_file: Path to SUMO network file (.net.xml)
            route_file: Path to SUMO route file (.rou.xml)
            reward_fn: Reward function instance
            use_gui: Whether to use SUMO GUI
            episode_length: Episode duration in seconds
            step_size: Simulation step size in seconds
            yellow_time: Yellow phase duration
            min_green: Minimum green phase duration
            max_green: Maximum green phase duration
            sumo_seed: Random seed for SUMO
            sumo_warnings: Whether to show SUMO warnings
        """
        super().__init__()
        
        self.net_file = net_file
        self.route_file = route_file
        self.reward_fn = reward_fn
        self.use_gui = use_gui
        self.episode_length = episode_length
        self.step_size = step_size
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.sumo_seed = sumo_seed
        self.sumo_warnings = sumo_warnings
        
        # SUMO connection
        self.sumo = None
        self.label = f"sumo_{id(self)}"
        
        # Load network
        self.net = sumolib.net.readNet(net_file)
        
        # Get traffic light IDs
        self.tls_objects = self.net.getTrafficLights()
        if not self.tls_objects:
            raise ValueError("No traffic lights found in network")
        self.tls_ids = [tls.getID() for tls in self.tls_objects]
        self.tls_ids = [tls.getID() for tls in self.tls_objects]
        
        # For simplicity, control first traffic light
        # Can be extended to multi-agent
        self.tls_id = self.tls_ids[0]
        
        # Get phases for this traffic light
        tls = self.tls_objects[0]
        self.phases = list(tls.getPrograms().values())[0].getPhases()
        
        # Filter out yellow/red phases (only green phases are actions)
        self.green_phases = [
            i for i, phase in enumerate(self.phases)
            if 'y' not in phase.state.lower() and 'r' in phase.state.lower()
        ]
        
        if not self.green_phases:
            # If no clear green phases, use all phases
            self.green_phases = list(range(len(self.phases)))
        
        # Get controlled lanes
        self.lanes = list(tls.getConnections().keys())
        
        # State and action spaces
        # State: [queue_lengths (per lane), waiting_times (per lane), current_phase, time_since_last_change]
        state_size = len(self.lanes) * 2 + 2
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(len(self.green_phases))
        
        # Episode state
        self.current_step = 0
        self.current_phase = 0
        self.time_since_phase_change = 0
        self.total_reward = 0.0
        
        print(f"SUMO Environment initialized:")
        print(f"  Traffic lights: {len(self.tls_ids)}")
        print(f"  Controlled TLS: {self.tls_id}")
        print(f"  Lanes: {len(self.lanes)}")
        print(f"  Actions (green phases): {len(self.green_phases)}")
        print(f"  State size: {state_size}")
    
    def _start_sumo(self):
        """Start SUMO simulation"""
        sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'
        
        sumo_cmd = [
            sumo_binary,
            '-n', self.net_file,
            '-r', self.route_file,
            '--step-length', str(self.step_size),
            '--waiting-time-memory', '1000',
            '--time-to-teleport', '-1',
            '--no-step-log', 'true',
            '--no-warnings', str(not self.sumo_warnings).lower(),
        ]
        
        if self.sumo_seed is not None:
            sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        
        # Start SUMO
        traci.start(sumo_cmd, label=self.label)
        self.sumo = traci.getConnection(self.label)
        
        # Set initial phase
        self.sumo.trafficlight.setPhase(self.tls_id, self.current_phase)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Close existing connection
        if self.sumo is not None:
            try:
                self.sumo.close()
            except:
                pass
        
        # Start new SUMO instance
        self._start_sumo()
        
        # Reset episode state
        self.current_step = 0
        self.current_phase = 0
        self.time_since_phase_change = 0
        self.total_reward = 0.0
        
        # Reset reward function
        self.reward_fn.reset()
        
        # Get initial state
        state = self._get_state()
        info = {'step': self.current_step}
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.
        
        Args:
            action: Index of green phase to activate
            
        Returns:
            state: Next state
            reward: Reward value
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Convert action to phase index
        target_phase = self.green_phases[action]
        
        # Check if phase change is needed
        phase_changed = False
        if target_phase != self.current_phase:
            # Only change if minimum green time has passed
            if self.time_since_phase_change >= self.min_green:
                # Insert yellow phase transition
                self._set_yellow_phase(self.current_phase)
                for _ in range(self.yellow_time):
                    self.sumo.simulationStep()
                
                # Set new phase
                self.sumo.trafficlight.setPhase(self.tls_id, target_phase)
                self.current_phase = target_phase
                self.time_since_phase_change = 0
                phase_changed = True
        
        # Simulate for step_size seconds
        for _ in range(self.step_size):
            self.sumo.simulationStep()
            self.time_since_phase_change += 1
        
        self.current_step += self.step_size
        
        # Get state information
        state_info = self._collect_state_info()
        
        # Compute reward
        reward, reward_info = self.reward_fn.compute_reward(state_info)
        self.total_reward += reward
        
        # Get next state
        next_state = self._get_state()
        
        # Check if episode is done
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # Compile info
        info = {
            'step': self.current_step,
            'phase': self.current_phase,
            'phase_changed': phase_changed,
            'time_since_phase_change': self.time_since_phase_change,
            'total_reward': self.total_reward,
            **reward_info
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        State includes:
        - Queue length per lane
        - Average waiting time per lane
        - Current phase (one-hot)
        - Time since last phase change
        """
        queue_lengths = []
        waiting_times = []
        
        for lane_id in self.lanes:
            # Queue length (number of halting vehicles)
            queue_length = self.sumo.lane.getLastStepHaltingNumber(lane_id)
            queue_lengths.append(queue_length)
            
            # Average waiting time
            vehicle_ids = self.sumo.lane.getLastStepVehicleIDs(lane_id)
            if vehicle_ids:
                lane_waiting_time = np.mean([
                    self.sumo.vehicle.getWaitingTime(vid) 
                    for vid in vehicle_ids
                ])
            else:
                lane_waiting_time = 0.0
            waiting_times.append(lane_waiting_time)
        
        # Normalize time since phase change
        normalized_time = min(self.time_since_phase_change / self.max_green, 1.0)
        
        # Combine into state vector
        state = np.array(
            queue_lengths + waiting_times + [self.current_phase, normalized_time],
            dtype=np.float32
        )
        
        return state
    
    def _collect_state_info(self) -> Dict:
        """
        Collect detailed state information for reward computation.
        """
        all_waiting_times = []
        all_queue_lengths = []
        all_vehicle_ids = []
        
        # Collect from all lanes
        for lane_id in self.lanes:
            vehicle_ids = self.sumo.lane.getLastStepVehicleIDs(lane_id)
            all_vehicle_ids.extend(vehicle_ids)
            
            for vid in vehicle_ids:
                waiting_time = self.sumo.vehicle.getWaitingTime(vid)
                all_waiting_times.append(waiting_time)
            
            queue_length = self.sumo.lane.getLastStepHaltingNumber(lane_id)
            all_queue_lengths.append(queue_length)
        
        return {
            'waiting_times': all_waiting_times if all_waiting_times else [0.0],
            'queue_lengths': all_queue_lengths if all_queue_lengths else [0.0],
            'vehicle_ids': all_vehicle_ids
        }
    
    def _set_yellow_phase(self, current_phase: int):
        """Set yellow phase for transition"""
        # Get current phase state
        phase_state = self.phases[current_phase].state
        
        # Convert to yellow (replace 'G' with 'y', keep 'r')
        yellow_state = phase_state.replace('G', 'y').replace('g', 'y')
        
        # Set yellow phase
        self.sumo.trafficlight.setRedYellowGreenState(self.tls_id, yellow_state)
    
    def render(self):
        """Render is handled by SUMO GUI"""
        pass
    
    def close(self):
        """Close SUMO connection"""
        if self.sumo is not None:
            try:
                self.sumo.close()
            except:
                pass
            self.sumo = None
    
    def get_metrics(self) -> Dict:
        """Get current episode metrics"""
        return self.reward_fn.get_episode_metrics()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()


class MultiIntersectionEnv(SumoTrafficEnv):
    """
    Extended environment for multiple intersections.
    Can be used for centralized or multi-agent control.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override to control all traffic lights
        self.tls_objects = self.net.getTrafficLights()
        
        # Each TLS gets its own action space
        # For centralized: joint action space
        # For multi-agent: would need separate policies
        
        print(f"Multi-intersection environment with {len(self.tls_ids)} traffic lights")


if __name__ == "__main__":
    # Test environment
    from reward import FairnessAwareReward, RewardConfig
    
    config = RewardConfig()
    reward_fn = FairnessAwareReward(config)
    
    env = SumoTrafficEnv(
        net_file="data/sumo_networks/single/network.net.xml",
        route_file="data/sumo_networks/single/routes.rou.xml",
        reward_fn=reward_fn,
        use_gui=False,
        episode_length=360,  # 6 minutes for testing
    )
    
    print("\nTesting environment...")
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    
    # Run a few steps
    for i in range(10):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, vehicles={info['num_vehicles']}")
        
        if terminated:
            break
    
    env.close()
    print("\nEnvironment test complete!")
