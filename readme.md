## Quick Start Guide

### 1. Setup on GPU Server

```bash
# Clone repository
git clone https://github.com/yourusername/fairness-traffic-control.git
cd fairness-traffic-control

# Create conda environment
conda env create -f environment.yml
conda activate fairness-tsc

# Install SUMO (headless)
wget https://sourceforge.net/projects/sumo/files/sumo/version-1.19.0/sumo-1.19.0.tar.gz
tar xzf sumo-1.19.0.tar.gz
cd sumo-1.19.0
./configure --prefix=$HOME/sumo --with-python
make -j8 && make install
echo 'export SUMO_HOME=$HOME/sumo/share/sumo' >> ~/.bashrc
echo 'export PATH=$HOME/sumo/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
cd ..

# Install project
pip install -e .

# Generate traffic scenarios
python scripts/generate_traffic.py --network single_intersection
python scripts/generate_traffic.py --network grid_4x4

# Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import traci; print('SUMO OK')"
```

### 2. Start Training

```bash
# Use tmux for persistent session
tmux new -s training

# Start training with TensorBoard
tensorboard --logdir results/logs --port 6006 --bind_all &

# Train model
python scripts/train.py \
    --config config/train_config.yaml \
    --network single_intersection \
    --algorithm DQN \
    --episodes 5000 \
    --gpu 0,1 \
    --name fairness_baseline

# Detach from tmux: Ctrl+B, then D
# Reattach: tmux attach -t training
```

### 3. Monitor Training (from your laptop)

```bash
# Create SSH tunnel for TensorBoard
ssh -L 6006:localhost:6006 user@gpu-server

# Access TensorBoard at: http://localhost:6006

# Or monitor with Jupyter
ssh -L 8888:localhost:8888 user@gpu-server
# Then open: http://localhost:8888
```

### 4. Download Models and Run Inference

```bash
# On server: Package trained models
tar -czf trained_models.tar.gz models/ config/ results/

# On laptop: Download and setup
scp user@gpu-server:~/fairness-traffic-control/trained_models.tar.gz .
tar -xzf trained_models.tar.gz

# Create CPU environment
conda create -n tsc-inference python=3.10 -y
conda activate tsc-inference
pip install -r requirements-inference.txt

# Install SUMO with GUI (Ubuntu)
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-gui

# Run inference with visualization
python scripts/inference.py \
    --model models/best_model.pth \
    --network data/sumo_networks/single/ \
    --episodes 10 \
    --visualize \
    --save-video results/demo.mp4
```

## File Checklist

Before starting, ensure you have:
- [ ] `environment.yml` - Conda environment
- [ ] `requirements.txt` - Python dependencies
- [ ] `config/train_config.yaml` - Training configuration
- [ ] `src/environment/reward.py` - Reward function
- [ ] `scripts/train.py` - Training script
- [ ] `scripts/generate_traffic.py` - Traffic generation
- [ ] `.gitignore` - Version control exclusions

## Expected Disk Usage

- SUMO installation: ~500 MB
- Conda environment: ~5 GB
- Project code: ~50 MB
- Training data: ~1-2 GB
- Model checkpoints: ~100 MB per checkpoint
- Results/logs: ~500 MB per experiment

**Total: ~10-15 GB**

## Troubleshooting

### SUMO not found
```bash
export SUMO_HOME=$HOME/sumo/share/sumo
export PATH=$HOME/sumo/bin:$PATH
```

### CUDA out of memory
```bash
# Reduce batch size
python scripts/train.py --batch-size 64

# Use single GPU
python scripts/train.py --gpu 0
```

### SSH disconnection
```bash
# Use tmux
tmux new -s training
python scripts/train.py ...
# Ctrl+B then D to detach
```