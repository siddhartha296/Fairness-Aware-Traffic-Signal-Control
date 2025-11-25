"""
Traffic Generation Script - FIXED VERSION
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

def get_sumo_tool(tool_name):
    """Check if SUMO tool is in PATH or SUMO_HOME"""
    if os.getenv("SUMO_HOME"):
        tool_path = Path(os.getenv("SUMO_HOME")) / ('tools' if tool_name.endswith('.py') else 'bin') / tool_name
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
    """Generates a simple 4-way single intersection network WITH TRAFFIC LIGHTS"""
    output_dir.mkdir(parents=True, exist_ok=True)
    net_file = output_dir / "network.net.xml"
    
    print(f"Generating single intersection network at {net_file}...")
    
    # FIXED: Use grid with proper TLS settings
    netgen_cmd = [
        get_sumo_tool("netgenerate"),
        "--grid",
        "--grid.number=1",           # Single intersection (1x1 grid)
        "--grid.length=300",          # Edge length
        "--grid.attach-length=300",   # Approach length
        "--default.lanenumber=2",     # 2 lanes per direction
        "--default.speed=13.89",      # Speed limit (50 km/h = 13.89 m/s)
        "--tls.guess=true",           # CRITICAL: Guess traffic lights at junctions
        "--tls.default-type=static",  # Use static traffic lights
        "--output-file", str(net_file)
    ]
    
    try:
        result = subprocess.run(netgen_cmd, check=True, capture_output=True, text=True)
        print("✓ Network file generated.")
        
        # Verify traffic lights were created
        with open(net_file, 'r') as f:
            content = f.read()
            if '<tlLogic' not in content:
                print("⚠️  WARNING: No traffic lights found in generated network!")
                print("Attempting to add traffic lights manually...")
                
                # Generate with explicit junction
                netgen_cmd_v2 = [
                    get_sumo_tool("netgenerate"),
                    "--grid",
                    "--grid.number=1",
                    "--grid.length=300",
                    "--grid.attach-length=300",
                    "--default.lanenumber=2",
                    "--default.speed=13.89",
                    "--junctions.join=false",  # Don't join junctions
                    "--tls.guess=true",
                    "--tls.default-type=static",
                    "--output-file", str(net_file)
                ]
                subprocess.run(netgen_cmd_v2, check=True, capture_output=True)
                
                # Verify again
                with open(net_file, 'r') as f:
                    if '<tlLogic' in f.read():
                        print("✓ Traffic lights added successfully")
                    else:
                        print("✗ Failed to add traffic lights")
                        print("Trying alternative method...")
                        generate_manual_intersection(output_dir)
                        return output_dir / "network.net.xml"
            else:
                print("✓ Traffic lights confirmed in network")
                
    except subprocess.CalledProcessError as e:
        print(f"Error generating network: {e.stderr}")
        sys.exit(1)
        
    return net_file

def generate_manual_intersection(output_dir: Path):
    """Generate intersection manually using node and edge files"""
    print("Generating intersection manually...")
    
    node_file = output_dir / "nodes.nod.xml"
    edge_file = output_dir / "edges.edg.xml"
    connection_file = output_dir / "connections.con.xml"
    net_file = output_dir / "network.net.xml"
    
    # Create nodes file
    nodes_content = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <node id="center" x="0.0" y="0.0" type="traffic_light"/>
    <node id="north" x="0.0" y="300.0" type="priority"/>
    <node id="south" x="0.0" y="-300.0" type="priority"/>
    <node id="east" x="300.0" y="0.0" type="priority"/>
    <node id="west" x="-300.0" y="0.0" type="priority"/>
</nodes>
"""
    
    # Create edges file
    edges_content = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <!-- Incoming edges -->
    <edge id="north_in" from="north" to="center" numLanes="2" speed="13.89"/>
    <edge id="south_in" from="south" to="center" numLanes="2" speed="13.89"/>
    <edge id="east_in" from="east" to="center" numLanes="2" speed="13.89"/>
    <edge id="west_in" from="west" to="center" numLanes="2" speed="13.89"/>
    
    <!-- Outgoing edges -->
    <edge id="north_out" from="center" to="north" numLanes="2" speed="13.89"/>
    <edge id="south_out" from="center" to="south" numLanes="2" speed="13.89"/>
    <edge id="east_out" from="center" to="east" numLanes="2" speed="13.89"/>
    <edge id="west_out" from="center" to="west" numLanes="2" speed="13.89"/>
</edges>
"""
    
    with open(node_file, 'w') as f:
        f.write(nodes_content)
    with open(edge_file, 'w') as f:
        f.write(edges_content)
    
    # Generate network from nodes and edges
    netconvert_cmd = [
        get_sumo_tool("netconvert"),
        "--node-files", str(node_file),
        "--edge-files", str(edge_file),
        "--output-file", str(net_file),
        "--tls.guess-signals=true",
        "--tls.default-type=static"
    ]
    
    try:
        subprocess.run(netconvert_cmd, check=True, capture_output=True)
        print("✓ Manual network generated with traffic lights")
        
        # Verify
        with open(net_file, 'r') as f:
            if '<tlLogic' in f.read():
                print("✓ Traffic lights confirmed")
            else:
                print("✗ Still no traffic lights - SUMO configuration issue")
                sys.exit(1)
                
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        sys.exit(1)

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
        sys.executable,
        get_sumo_tool("randomTrips.py"),
        "-n", str(network_file),
        "-o", str(trip_file),
        "-e", str(duration),
        "-p", str(period),
    ]
    
    # 2. Generate routes from trips
    duarouter_cmd = [
        get_sumo_tool("duarouter"),
        "-n", str(network_file),
        "-t", str(trip_file),
        "-o", str(route_file),
        "--ignore-errors",
    ]
    
    # 3. Create SUMO config file
    config_file = output_dir / "scenario.sumocfg"
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
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
        if trip_file.exists():
            trip_file.unlink()
        
        print("✓ Traffic generated successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error generating traffic: {e.stderr}")
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
            
            # Copy net file to demand directory
            import shutil
            demand_net_file = demand_dir / net_file.name
            shutil.copy2(net_file, demand_net_file)
            
            generate_traffic(
                network_file=demand_net_file,
                output_dir=demand_dir,
                duration=args.duration,
                demand=demand.strip()
            )
            
    elif args.network == 'grid_4x4':
        print("Grid network generation not implemented yet.")
        
    else:
        print(f"Unknown network type: {args.network}")
        sys.exit(1)
        
    print("\n" + "="*60)
    print("All scenarios generated successfully!")
    print("="*60)
    print("\nVerify traffic lights:")
    print(f"  grep '<tlLogic' {network_dir / 'network.net.xml'}")
    print("\nNext steps:")
    print("  1. Verify setup: python scripts/verify_setup.py")
    print("  2. Test training: python scripts/train.py --episodes 10 --debug --gpu 0")

if __name__ == "__main__":
    main()