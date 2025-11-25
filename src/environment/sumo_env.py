"""
SUMO Traffic Environment Wrapper - FIXED VERSION
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
        self.tls_list = list(self.net.getTrafficLights())
        if not self.tls_list:
            raise ValueError("No traffic lights found in network")
        
        # For simplicity, control first traffic light
        # Can be extended to multi-agent
        self.tls = self.tls_list[0]
        self.tls_id = self.tls.getID()
        
        print(f"Controlling traffic light: {self.tls_id}")
        
        # Get controlled lanes - will be determined at runtime
        self.lanes = []
        self.num_phases = 4  # Default, will be updated after SUMO starts
        
        # Action and observation spaces will be set after initialization
        # For now, set temporary values that will be updated
        self.action_space = None
        self.observation_space = None
        
        # Episode state
        self.current_step = 0
        self.current_phase = 0
        self.time_since_phase_change = 0
        self.total_reward = 0.0
        
        # Will be initialized after first SUMO connection
        self._initialized = False
    
    def _initialize_after_sumo_start(self):
        """Initialize environment details after SUMO has started"""
        if self._initialized:
            return
        
        # Get actual controlled lanes from TraCI
        self.lanes = list(self.sumo.trafficlight.getControlledLanes(self.tls_id))
        # Remove duplicates
        self.lanes = list(set(self.lanes))
        
        # Get traffic light program
        logic = self.sumo.trafficlight.getAllProgramLogics(self.tls_id)[0]
        self.phases = logic.phases
        
        # Create list of distinct phase states for control
        # We'll use setRedYellowGreenState instead of setPhase for more control
        self.phase_states = []
        for phase in self.phases:
            state = phase.state
            # Only include states with green lights
            if 'G' in state or 'g' in state:
                if state not in self.phase_states:
                    self.phase_states.append(state)
        
        if not self.phase_states:
            # Fallback: use all phase states
            self.phase_states = [phase.state for phase in self.phases]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_states = []
        for state in self.phase_states:
            if state not in seen:
                seen.add(state)
                unique_states.append(state)
        self.phase_states = unique_states
        
        self.num_phases = len(self.phase_states)
        
        # Update action space
        self.action_space = spaces.Discrete(self.num_phases)
        
        # Update observation space
        state_size = len(self.lanes) * 2 + 2  # queue + wait per lane + phase + time
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        
        print(f"Environment initialized:")
        print(f"  Traffic light: {self.tls_id}")
        print(f"  Controlled lanes: {len(self.lanes)}")
        print(f"  Total phases: {len(self.phases)}")
        print(f"  Unique phase states: {self.num_phases}")
        print(f"  Phase states: {self.phase_states}")
        print(f"  Actions: {self.num_phases}")
        print(f"  State size: {state_size}")
        
        self._initialized = True
    
    def _start_sumo(self):
        """Start SUMO simulation"""
        sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'
        
        sumo_cmd = [
            sumo_binary,
            '-n', self.net_file,
            '-r', self.route_file,
            '--step-length', str(1),  # Always use 1 second steps internally
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
        
        # Initialize after SUMO starts
        self._initialize_after_sumo_start()
        
        # Set initial phase state
        if self.phase_states:
            self.sumo.trafficlight.setRedYellowGreenState(self.tls_id, self.phase_states[0])
    
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
        
        # Run simulation for a few steps to populate vehicles
        for _ in range(10):
            self.sumo.simulationStep()
        
        # Get initial state
        state = self._get_state()
        info = {'step': self.current_step}
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.
        
        Args:
            action: Index of phase state to activate
            
        Returns:
            state: Next state
            reward: Reward value
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Validate action is in range
        if action < 0 or action >= len(self.phase_states):
            action = 0  # Default to first action if out of range
        
        # Get target phase state
        target_state = self.phase_states[action]
        current_state = self.sumo.trafficlight.getRedYellowGreenState(self.tls_id)
        
        # Check if phase change is needed
        phase_changed = False
        if target_state != current_state:
            # Only change if minimum green time has passed
            if self.time_since_phase_change >= self.min_green:
                # Insert yellow phase transition
                yellow_state = current_state.replace('G', 'y').replace('g', 'y')
                self.sumo.trafficlight.setRedYellowGreenState(self.tls_id, yellow_state)
                
                for _ in range(self.yellow_time):
                    self.sumo.simulationStep()
                
                # Set new phase state
                self.sumo.trafficlight.setRedYellowGreenState(self.tls_id, target_state)
                self.current_phase = action
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
        - Current phase (normalized)
        - Time since last phase change (normalized)
        """
        queue_lengths = []
        waiting_times = []
        
        for lane_id in self.lanes:
            # Queue length (number of halting vehicles)
            try:
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
            except:
                # Lane might not exist yet or error
                queue_lengths.append(0.0)
                waiting_times.append(0.0)
        
        # Normalize current phase
        normalized_phase = self.current_phase / max(len(self.phases) - 1, 1)
        
        # Normalize time since phase change
        normalized_time = min(self.time_since_phase_change / self.max_green, 1.0)
        
        # Combine into state vector
        state = np.array(
            queue_lengths + waiting_times + [normalized_phase, normalized_time],
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
            try:
                vehicle_ids = self.sumo.lane.getLastStepVehicleIDs(lane_id)
                all_vehicle_ids.extend(vehicle_ids)
                
                for vid in vehicle_ids:
                    waiting_time = self.sumo.vehicle.getWaitingTime(vid)
                    all_waiting_times.append(waiting_time)
                
                queue_length = self.sumo.lane.getLastStepHaltingNumber(lane_id)
                all_queue_lengths.append(queue_length)
            except:
                pass
        
        return {
            'waiting_times': all_waiting_times if all_waiting_times else [0.0],
            'queue_lengths': all_queue_lengths if all_queue_lengths else [0.0],
            'vehicle_ids': all_vehicle_ids
        }
    
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


if __name__ == "__main__":
    # Test environment
    from reward import FairnessAwareReward, RewardConfig
    
    config = RewardConfig()
    reward_fn = FairnessAwareReward(config)
    
    network_path = "data/sumo_networks/single_intersection/medium"
    
    env = SumoTrafficEnv(
        net_file=f"{network_path}/network.net.xml",
        route_file=f"{network_path}/routes.rou.xml",
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
        print(f"Step {i}: reward={reward:.3f}, vehicles={info.get('num_vehicles', 0)}")
        
        if terminated:
            break
    
    env.close()
    print("\nEnvironment test complete!")