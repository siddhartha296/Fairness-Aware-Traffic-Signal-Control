#!/usr/bin/env python3
"""
Setup Verification Script
scripts/verify_setup.py

Verifies that all components are ready for training.
Run this before starting training to catch any issues early.
"""

import sys
import os
from pathlib import Path

def check_python_packages():
    """Check if required Python packages are installed"""
    print("\n" + "="*60)
    print("1. Checking Python Packages")
    print("="*60)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('gymnasium', 'Gymnasium'),
        ('yaml', 'PyYAML'),
        ('traci', 'TraCI (SUMO)'),
        ('sumolib', 'SUMO Library'),
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_cuda():
    """Check CUDA availability"""
    print("\n" + "="*60)
    print("2. Checking CUDA/GPU")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠️  CUDA not available - will use CPU")
            print("  Training will be slower but should work")
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False
    
    return True

def check_sumo():
    """Check SUMO installation"""
    print("\n" + "="*60)
    print("3. Checking SUMO")
    print("="*60)
    
    sumo_home = os.getenv('SUMO_HOME')
    if sumo_home:
        print(f"  ✓ SUMO_HOME: {sumo_home}")
    else:
        print("  ⚠️  SUMO_HOME not set")
        print("  Set with: export SUMO_HOME=$HOME/sumo/share/sumo")
    
    # Check if sumo binary exists
    try:
        import subprocess
        result = subprocess.run(['sumo', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            print(f"  ✓ SUMO version: {version}")
        else:
            print("  ✗ Could not run 'sumo' command")
            return False
    except FileNotFoundError:
        print("  ✗ 'sumo' command not found in PATH")
        print("  Add to PATH: export PATH=$HOME/sumo/bin:$PATH")
        return False
    except Exception as e:
        print(f"  ✗ Error checking SUMO: {e}")
        return False
    
    return True

def check_project_structure():
    """Check if project structure is correct"""
    print("\n" + "="*60)
    print("4. Checking Project Structure")
    print("="*60)
    
    required_dirs = [
        'src',
        'src/environment',
        'src/models',
        'src/training',
        'src/utils',
        'config',
        'scripts',
    ]
    
    required_files = [
        'src/environment/sumo_env.py',
        'src/environment/reward.py',
        'src/models/dqn.py',
        'src/training/trainer.py',
        'src/utils/logger.py',
        'src/utils/checkpoint.py',
        'config/train_config.yaml',
        'scripts/train.py',
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - MISSING")
            all_ok = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_ok = False
    
    return all_ok

def check_traffic_data():
    """Check if traffic scenarios are generated"""
    print("\n" + "="*60)
    print("5. Checking Traffic Scenarios")
    print("="*60)
    
    networks_dir = Path('data/sumo_networks')
    
    if not networks_dir.exists():
        print(f"  ✗ {networks_dir} not found")
        print("  Generate with: python scripts/generate_traffic.py --network single_intersection")
        return False
    
    # Check for single_intersection scenarios
    single_dir = networks_dir / 'single_intersection'
    if single_dir.exists():
        print(f"  ✓ Found single_intersection network")
        
        demands = ['low', 'medium', 'high', 'rush_hour']
        for demand in demands:
            demand_dir = single_dir / demand
            if demand_dir.exists():
                net_file = demand_dir / 'network.net.xml'
                route_file = demand_dir / 'routes.rou.xml'
                config_file = demand_dir / 'scenario.sumocfg'
                
                if net_file.exists() and route_file.exists() and config_file.exists():
                    print(f"    ✓ {demand} scenario complete")
                else:
                    print(f"    ✗ {demand} scenario incomplete")
                    return False
            else:
                print(f"    ✗ {demand} scenario not found")
                return False
    else:
        print(f"  ✗ single_intersection network not found")
        print("  Generate with: python scripts/generate_traffic.py --network single_intersection")
        return False
    
    return True

def check_config():
    """Check if config file is valid"""
    print("\n" + "="*60)
    print("6. Checking Configuration")
    print("="*60)
    
    config_file = Path('config/train_config.yaml')
    
    if not config_file.exists():
        print(f"  ✗ Config file not found: {config_file}")
        return False
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['environment', 'reward', 'model', 'training', 'hardware']
        for key in required_keys:
            if key in config:
                print(f"  ✓ {key} section present")
            else:
                print(f"  ✗ {key} section missing")
                return False
        
    except Exception as e:
        print(f"  ✗ Error parsing config: {e}")
        return False
    
    return True

def main():
    print("\n" + "="*70)
    print(" FAIRNESS-AWARE TRAFFIC SIGNAL CONTROL - SETUP VERIFICATION")
    print("="*70)
    
    checks = [
        ("Python Packages", check_python_packages),
        ("CUDA/GPU", check_cuda),
        ("SUMO", check_sumo),
        ("Project Structure", check_project_structure),
        ("Traffic Scenarios", check_traffic_data),
        ("Configuration", check_config),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print(" VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8s} {name}")
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! 🎉")
        print("\nYou're ready to start training!")
        print("\nQuick start commands:")
        print("  # Test run (10 episodes, fast)")
        print("  python scripts/train.py --config config/train_config.yaml \\")
        print("      --network single_intersection --algorithm DQN \\")
        print("      --episodes 10 --batch-size 64 --gpu 0 --debug")
        print()
        print("  # Full training")
        print("  python scripts/train.py --config config/train_config.yaml \\")
        print("      --network single_intersection --algorithm DQN \\")
        print("      --episodes 5000 --batch-size 128 --gpu 0")
        print()
        print("  # Monitor with TensorBoard")
        print("  tensorboard --logdir results/logs --port 6006 --bind_all")
        return 0
    else:
        print("\n❌ SETUP INCOMPLETE")
        print("\nPlease fix the issues above before training.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
