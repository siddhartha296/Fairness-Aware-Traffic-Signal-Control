"""
Traffic Generation Script
scripts/generate_traffic.py

Generates SUMO network files (.net.xml) and route files (.rou.xml)
for different scenarios.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Default SUMO tools (assumes SUMO is in your PATH)
NETGENERATE = "netgenerate"
RANDOM_TRIPS = "randomTrips.py"
DUAROUTER = "duarouter"
SUMO = "sumo"

def get_sumo_tool(tool_name):
    """Check if SUMO tool is in PATH or SUMO_HOME"""
    if os.getenv("SUMO_HOME"):
        tool_path = Path(os.getenv("SUMO_HOME")) / "bin" / tool_name
        if tool_path.exists():
            return str(tool_path)
    
    # Check in PATH
    try:
        subprocess.run([tool_name, "--version"], capture_output=True, check=True)
        return tool_name
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Error: '{tool_name}' not found.")
        print("Please ensure SUMO is installed and 'SUMO_HOME/bin' is in your PATH.")
        sys.exit(1)

def generate_single_intersection(output_dir: Path):
    """Generates a simple 4-way single intersection network"""
    output_dir.mkdir(parents=True, exist_ok=True)
    net_file = output_dir / "network.net.xml"
    
    print(f"Generating single intersection network at {net_file}...")
    
    # Basic netgenerate command
    netgen_cmd = [
        get_sumo_tool(NETGENERATE),
        "--grid",
        "--grid.number=1",
        "--grid.length=500",
        "-L=4", # Number of lanes
        "--tls.default-type=static",
        "--tls.layout=opposite",
        "--output-file", str(net_file)
    ]
    
    try:
        subprocess.run(netgen_cmd, check=True, capture_output=True)
        print("✓ Network file generated.")
    except subprocess.CalledProcessError as e:
        print(f"Error generating network: {e.stderr.decode()}")
        sys.exit(1)
        
    return net_file

def generate_traffic(network_file: Path, output_dir: Path, duration: int, demand: str):
    """Generates route file for a given demand level"""
    
    # Map demand to vehicles per hour
    demand_map = {
        'low': 300,
        'medium': 600,
        'high': 1000,
        'rush_hour': 1500
    }
    
    if demand not in demand_map:
        raise ValueError(f"Unknown demand level: {demand}. Choose from {list(demand_map.keys())}")
    
    vehicles_per_hour = demand_map[demand]
    period = 3600.0 / vehicles_per_hour # Time between vehicles
    
    route_file = output_dir / "routes.rou.xml"
    trip_file = output_dir / "trips.trip.xml"
    
    print(f"Generating traffic for demand '{demand}' ({vehicles_per_hour} veh/hr)...")
    
    # 1. Generate random trips
    random_trips_cmd = [
        sys.executable, # Use current python interpreter
        get_sumo_tool(RANDOM_TRIPS),
        "-n", str(network_file),
        "-o", str(trip_file),
        "-e", str(duration),
        "-p", str(period),
        "--validate"
    ]
    
    # 2. Generate routes from trips
    duarouter_cmd = [
        get_sumo_tool(DUAROUTER),
        "-n", str(network_file),
        "-t", str(trip_file),
        "-o", str(route_file),
        "--ignore-errors",
        "--validate"
    ]
    
    # 3. Create SUMO config file
    config_file = output_dir / "scenario.sumocfg"
    config_content = f"""
<configuration>
    <input>
        <net-file value="{network_file.name}"/>
        <route-files value="{route_file.name}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{duration}"/>
    </time>
    <processing>
        <waiting-time-memory value="1000"/>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>
"""
    
    try:
        print("  Running randomTrips.py...")
        subprocess.run(random_trips_cmd, check=True, capture_output=True)
        
        print("  Running duarouter...")
        subprocess.run(duarouter_cmd, check=True, capture_output=True)
        
        print(f"  Writing config file {config_file}...")
        with open(config_file, 'w') as f:
            f.write(config_content)
            
        # Cleanup trip file
        trip_file.unlink()
        
        print("✓ Traffic generated successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error generating traffic: {e.stderr.decode()}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Generate SUMO traffic scenarios'
    )
    
    parser.add_argument(
        '--network',
        type=str,
        default='single_intersection',
        choices=['single_intersection', 'grid_4x4'],
        help='Network type to generate'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Duration of the simulation in seconds'
    )
    parser.add_argument(
        '--demand',
        type=str,
        default='low,medium,high,rush_hour',
        help='Comma-separated list of demand levels to generate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/sumo_networks',
        help='Base output directory'
    )
    
    args = parser.parse_args()
    
    # Base output path
    base_output_dir = Path(args.output)
    
    if args.network == 'single_intersection':
        network_dir = base_output_dir / 'single_intersection'
        net_file = generate_single_intersection(network_dir)
        
        for demand in args.demand.split(','):
            demand_dir = network_dir / demand.strip()
            demand_dir.mkdir(parents=True, exist_ok=True)
            # Copy net file
            os.link(net_file, demand_dir / net_file.name)
            
            generate_traffic(
                network_file=demand_dir / net_file.name,
                output_dir=demand_dir,
                duration=args.duration,
                demand=demand.strip()
            )
            
    elif args.network == 'grid_4x4':
        print("Grid network generation not implemented yet.")
        # TODO: Add grid generation logic
        
    else:
        print(f"Unknown network type: {args.network}")
        sys.exit(1)
        
    print("\nAll scenarios generated!")

if __name__ == "__main__":
    main()
