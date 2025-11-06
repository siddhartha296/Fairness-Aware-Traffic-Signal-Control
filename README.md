# Fairness-Aware-Traffic-Signal-Control

# Fairness-Aware Traffic Signal Control with Reinforcement Learning

## Project Overview
Implementation of a novel RL-based traffic signal controller that optimizes for both efficiency and fairness, penalizing high variance in waiting times and prioritizing vehicles with exceptionally long waits.

---

## 📁 Project Structure

```
fairness-traffic-control/
├── README.md
├── environment.yml                 # Conda environment specification
├── requirements.txt               # Pip requirements
├── setup.py                       # Package setup
│
├── config/
│   ├── train_config.yaml         # Training configuration
│   ├── eval_config.yaml          # Evaluation configuration
│   └── network_configs/          # Traffic network definitions
│       ├── single_intersection.yaml
│       ├── grid_4x4.yaml
│       └── arterial.yaml
│
├── data/
│   ├── raw/                      # Raw traffic data
│   ├── processed/                # Processed datasets
│   └── sumo_networks/            # SUMO network files
│       ├── single/
│       ├── grid/
│       └── arterial/
│
├── src/
│   ├── __init__.py
│   │
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── sumo_env.py          # SUMO environment wrapper
│   │   ├── state.py             # State representation
│   │   └── reward.py            # Fairness-aware reward function
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dqn.py               # DQN implementation
│   │   ├── ppo.py               # PPO implementation
│   │   └── network.py           # Neural network architectures
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loop
│   │   ├── replay_buffer.py    # Experience replay
│   │   └── callbacks.py         # Training callbacks
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Fairness & efficiency metrics
│   │   └── visualizer.py        # Results visualization
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Logging utilities
│       ├── checkpoint.py        # Model checkpointing
│       └── data_loader.py       # Data loading utilities
│
├── scripts/
│   ├── generate_traffic.py      # Generate SUMO traffic scenarios
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   ├── inference.py             # Inference on laptop
│   └── visualize_results.py     # Results visualization
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_reward_analysis.ipynb
│   ├── 03_training_monitor.ipynb
│   └── 04_results_visualization.ipynb
│
├── tests/
│   ├── test_environment.py
│   ├── test_reward.py
│   └── test_models.py
│
├── models/                       # Saved model checkpoints
│   └── .gitkeep
│
└── results/                      # Training results & logs
    ├── logs/
    ├── checkpoints/
    └── visualizations/
```

---

## 🎯 Novel Reward Function Design

### Mathematical Formulation

```
R(t) = w_eff * R_efficiency(t) + w_fair * R_fairness(t) + w_penalty * R_penalty(t)

where:

1. R_efficiency(t) = -α * (avg_waiting_time + β * avg_queue_length)

2. R_fairness(t) = -γ * std_waiting_time - δ * max_waiting_time

3. R_penalty(t) = -Σ exp(λ * (w_i - threshold)) for all vehicles with w_i > threshold
```

### Key Innovation
- **Variance Penalty**: Penalizes high standard deviation in waiting times
- **Max Wait Penalty**: Exponentially penalizes exceptionally long waits
- **Adaptive Thresholds**: Dynamic threshold based on current traffic conditions

---

## 🛠️ Setup Instructions

### 1. Conda Environment Setup (GPU Server)

```bash
# Create environment from YAML
conda env create -f environment.yml
conda activate fairness-tsc

# Or create manually
conda create -n fairness-tsc python=3.10 -y
conda activate fairness-tsc

# Install PyTorch with CUDA support (for L40s GPUs)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### 2. SUMO Installation (Headless)

```bash
# Download SUMO (without GUI for server)
wget https://sourceforge.net/projects/sumo/files/sumo/version-1.19.0/sumo-1.19.0.tar.gz
tar xzf sumo-1.19.0.tar.gz
cd sumo-1.19.0

# Configure and install to user directory
./configure --prefix=$HOME/sumo --with-python
make -j8
make install

# Add to PATH
echo 'export SUMO_HOME=$HOME/sumo/share/sumo' >> ~/.bashrc
echo 'export PATH=$HOME/sumo/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. Project Installation

```bash
# Clone/upload project
cd fairness-traffic-control

# Install in development mode
pip install -e .

# Verify installation
python -c "import sumo; import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📊 Datasets

### 1. Synthetic Traffic Data
Generate using SUMO's traffic generation tools:

```bash
python scripts/generate_traffic.py \
    --network single_intersection \
    --duration 3600 \
    --demand low,medium,high,rush_hour \
    --output data/sumo_networks/single/
```

**Traffic Patterns:**
- Low demand: 100-300 vehicles/hour
- Medium demand: 500-800 vehicles/hour
- High demand: 1000-1500 vehicles/hour
- Rush hour: Peak patterns with time-varying demand

### 2. Real-World Datasets (Optional)

**RESCO Dataset** (Traffic signal control benchmark)
- Source: https://github.com/Pi-Star-Lab/RESCO
- Contains real-world intersection data
- Multiple cities and scenarios

**Download:**
```bash
mkdir -p data/raw/resco
cd data/raw/resco
wget https://github.com/Pi-Star-Lab/RESCO/archive/refs/heads/main.zip
unzip main.zip
```

### 3. Data Preprocessing

```bash
python scripts/preprocess_data.py \
    --input data/raw/ \
    --output data/processed/ \
    --validation-split 0.2
```

---

## 🚀 Training Workflow

### Phase 1: Environment Testing (Day 1)

```bash
# Test SUMO environment
python tests/test_environment.py

# Test reward function
python tests/test_reward.py

# Visualize reward components
jupyter notebook notebooks/02_reward_analysis.ipynb
```

### Phase 2: Small-Scale Training (Days 2-3)

```bash
# Single intersection, short episodes
python scripts/train.py \
    --config config/train_config.yaml \
    --network single_intersection \
    --algorithm DQN \
    --episodes 1000 \
    --batch-size 64 \
    --gpu 0
```

### Phase 3: Full Training (Days 4-7)

```bash
# Multi-GPU training for grid network
python scripts/train.py \
    --config config/train_config.yaml \
    --network grid_4x4 \
    --algorithm PPO \
    --episodes 10000 \
    --batch-size 256 \
    --gpu 0,1 \
    --checkpoint-freq 500
```

### Phase 4: Hyperparameter Tuning (Days 8-10)

```bash
# Grid search for reward weights
python scripts/train.py \
    --config config/train_config.yaml \
    --tune-hyperparams \
    --trials 50 \
    --gpu 0,1
```

---

## 📈 Monitoring Training (GPU Server)

### TensorBoard (Web-based, no GUI needed)

```bash
# Start TensorBoard on server
tensorboard --logdir results/logs --port 6006 --bind_all

# On your laptop, create SSH tunnel:
# ssh -L 6006:localhost:6006 user@gpu-server
# Then access: http://localhost:6006
```

### Jupyter Notebook Monitoring

```bash
# Start Jupyter on server
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# On laptop:
# ssh -L 8888:localhost:8888 user@gpu-server
# Access: http://localhost:8888
```

---

## 💻 Inference on Laptop

### 1. Download Trained Model

```bash
# On server
tar -czf trained_models.tar.gz models/ config/

# On laptop
scp user@gpu-server:~/fairness-traffic-control/trained_models.tar.gz .
tar -xzf trained_models.tar.gz
```

### 2. Setup Laptop Environment

```bash
# Create CPU-only environment
conda create -n tsc-inference python=3.10 -y
conda activate tsc-inference

# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install SUMO with GUI support
sudo apt-get install sumo sumo-tools sumo-gui  # Linux
# or brew install sumo  # macOS

pip install -r requirements-inference.txt
```

### 3. Run Inference

```bash
# Evaluate model with visualization
python scripts/inference.py \
    --model models/best_model.pth \
    --network data/sumo_networks/single/ \
    --episodes 10 \
    --visualize \
    --save-video results/inference_video.mp4
```

---

## 📋 Configuration Files

### environment.yml
```yaml
name: fairness-tsc
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch::pytorch>=2.0.0
  - pytorch::torchvision
  - pytorch::torchaudio
  - pytorch::pytorch-cuda=12.1
  - numpy>=1.24.0
  - scipy>=1.10.0
  - pandas>=2.0.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - tensorboard>=2.13.0
  - jupyter
  - pyyaml
  - tqdm
  - pip
  - pip:
      - gymnasium>=0.28.0
      - stable-baselines3>=2.0.0
      - sumolib
      - traci
      - wandb
      - opencv-python-headless
      - scikit-learn
```

### config/train_config.yaml
```yaml
# Training Configuration
experiment_name: "fairness_aware_tsc"
seed: 42

# Environment
environment:
  name: "SumoTrafficEnv"
  episode_length: 3600  # 1 hour
  step_size: 5  # seconds
  yellow_time: 3
  min_green: 10
  max_green: 60

# Reward Function
reward:
  type: "fairness_aware"
  weights:
    efficiency: 0.4
    fairness: 0.4
    penalty: 0.2
  parameters:
    alpha: 1.0        # waiting time weight
    beta: 0.5         # queue length weight
    gamma: 2.0        # std deviation weight
    delta: 1.5        # max wait weight
    lambda: 0.1       # exponential penalty rate
    threshold: 120    # seconds

# Model
model:
  algorithm: "DQN"  # or "PPO"
  architecture:
    hidden_layers: [256, 256, 128]
    activation: "relu"
  learning_rate: 0.0001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995

# Training
training:
  episodes: 10000
  batch_size: 128
  replay_buffer_size: 100000
  target_update_freq: 1000
  checkpoint_freq: 500
  eval_freq: 100
  num_eval_episodes: 10

# Hardware
hardware:
  gpu_ids: [0, 1]
  num_workers: 8
  prefetch_factor: 2
```

---

## 📊 Evaluation Metrics

### Efficiency Metrics
- Average waiting time
- Average queue length
- Throughput (vehicles/hour)
- Average delay

### Fairness Metrics
- **Waiting time variance**: σ²(waiting times)
- **Gini coefficient**: Inequality measure (0 = perfect equality)
- **Max waiting time**: 95th percentile and absolute max
- **Fairness index**: Jain's fairness index
- **Starvation count**: Vehicles waiting > threshold

### Comparison Baselines
1. Fixed-time controller
2. Max-pressure controller
3. Standard DQN (efficiency-only)
4. SOTL (Self-Organizing Traffic Lights)

---

## 📝 Expected Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Setup | 1 day | Environment setup, SUMO installation, data generation |
| Development | 2-3 days | Implement reward function, environment wrapper, model |
| Initial Training | 2-3 days | Single intersection experiments, debugging |
| Full Training | 4-5 days | Multi-intersection training with hyperparameter tuning |
| Evaluation | 2 days | Comprehensive evaluation, comparison with baselines |
| Inference Setup | 1 day | Setup laptop environment, transfer models |
| Documentation | 1-2 days | Write paper, create visualizations |

**Total: ~2 weeks**

---

## 🔧 Troubleshooting

### SUMO Connection Issues
```python
# In src/environment/sumo_env.py, use headless mode
sumo_cmd = ["sumo", "-c", config_file, "--no-warnings", "--no-step-log"]
```

### GPU Out of Memory
```bash
# Reduce batch size
python scripts/train.py --batch-size 64

# Enable gradient checkpointing
python scripts/train.py --gradient-checkpointing
```

### SSH Disconnection
```bash
# Use tmux for persistent sessions
tmux new -s training
python scripts/train.py ...
# Ctrl+B, then D to detach
# tmux attach -t training to reattach
```

---

## 📚 Key Papers to Cite

1. Reinforcement Learning for Traffic Signal Control
2. Fairness in Machine Learning: A Survey
3. SUMO: Simulation of Urban MObility
4. Deep Q-Networks (DQN) - Mnih et al.
5. Proximal Policy Optimization (PPO) - Schulman et al.

---

## 🎓 Deliverables

1. **Code Repository**: Complete implementation with documentation
2. **Trained Models**: Checkpoints for best performing models
3. **Evaluation Report**: Comprehensive metrics comparison
4. **Visualizations**: Training curves, fairness analysis, traffic animations
5. **Research Paper**: Novel reward function and experimental results

---

## 📬 Contact & Support

For issues or questions, create an issue in the repository or contact the project maintainer.

**Happy Training! 🚦🤖**