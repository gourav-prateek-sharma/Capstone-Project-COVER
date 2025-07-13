#!/usr/bin/env python3
# filepath: /home/neel/gpsharma/Capstone-Project-COVER/generate_rss_csv.py
import argparse
import os
import numpy as np
from sionna.rt import load_scene, scene as rt_scene
from coverage_helpers import rss_map_full, MAX_DEPTH, CELL_SIZE, SAMPLES_PER_TX
from scene_helpers import get_scene_bounds3d
import pandas as pd

def rss_write_csv(scene_obj, tx_position, max_depth=MAX_DEPTH, cell_size=CELL_SIZE, samples_per_tx=SAMPLES_PER_TX, csv_file="rss_output.csv", compress=False):
    """
    Generates the RSS tensor using rss_map_full and writes it to a CSV file.

    Args:
        scene_obj: Sionna RT scene object.
        tx_position (list or array): Transmitter position.
        max_depth (int): Maximum depth parameter for rss_map_full.
        cell_size (list): Cell size parameter for rss_map_full.
        samples_per_tx (int): Number of samples per transmitter.
        csv_file (str): Filename (with path) for the CSV output.
        compress (bool): Whether to compress the output using gzip.
        
    Returns:
        rss: The RSS tensor (as a numpy array).
    """
    # Generate the RSS using the existing function
    rss = rss_map_full(scene_obj, tx_position=tx_position, max_depth=max_depth,
                       cell_size=cell_size, samples_per_tx=samples_per_tx, csv_file=None)
    # Convert to numpy array if necessary
    if hasattr(rss, 'numpy'):
        rss_array = rss.numpy()
    else:
        rss_array = np.array(rss)

    # Ensure the RSS array is 2-D before writing to CSV; for example, squeeze shape (1,603,738) to (603,738)
    if rss_array.ndim != 2:
        rss_array = np.squeeze(rss_array)
        if rss_array.ndim != 2:
            raise ValueError(f"After squeezing, RSS array must have 2 dimensions. Current shape: {rss_array.shape}")

    # Write the RSS tensor to a CSV file
    df = pd.DataFrame(rss_array)
    if compress:
        # Add .gz extension if not already present
        if not csv_file.endswith('.gz'):
            csv_file = csv_file + '.gz'
        df.to_csv(csv_file, index=False, header=False, compression='gzip')
    else:
        df.to_csv(csv_file, index=False, header=False)
    
    return rss_array

def main():
    parser = argparse.ArgumentParser(description="Generate RSS CSV files for transmitter positions.")
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene name string, e.g., 'munich' (assumes attribute in sionna.rt.scene)")
    parser.add_argument("--tx_positions_file", type=str, default="",
                        help="Path to CSV file containing tx positions (one per line as x,y,z). Ignored if --N > 0.")
    parser.add_argument("--N", type=int, default=0,
                        help="Number of random transmitter positions to generate. If >0, overrides tx_positions_file.")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Output directory where the RSS CSV files will be stored")
    parser.add_argument("--compress", action="store_true",
                        help="Compress output files using gzip")
    args = parser.parse_args()

    # Create the output directory if it does not exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Load the scene using sionna.rt.load_scene. It is assumed that the scene string corresponds
    # to an attribute of sionna.rt.scene (e.g., sionna.rt.scene.munich)
    try:
        scene_attr = getattr(rt_scene, args.scene)
    except AttributeError:
        raise ValueError(f"Scene '{args.scene}' not found in sionna.rt.scene.")

    scene_obj = load_scene(scene_attr, merge_shapes=True)

    # Determine the tx_positions array: either generate randomly if N > 0 or load from file.
    if args.N > 0:
        # Get scene bounds
        x_min, x_max, y_min, y_max, z_min, z_max = get_scene_bounds3d(scene_obj)
        bounds = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
        low = bounds[:, 0]
        high = bounds[:, 1]
        tx_positions = np.random.uniform(low, high, size=(args.N, 3))
        print(f"Generated {args.N} random tx positions using scene bounds:")
        print(f"  x: [{x_min}, {x_max}], y: [{y_min}, {y_max}], z: [{z_min}, {z_max}]")
    else:
        if not args.tx_positions_file:
            raise ValueError("Either provide --tx_positions_file or set --N > 0 to generate random positions.")
        try:
            # Use np.loadtxt assuming the file has rows of 3 numbers
            tx_positions = np.loadtxt(args.tx_positions_file, delimiter=',')
        except Exception:
            # Fall back to whitespace delimiter in case csv isn't comma separated.
            tx_positions = np.loadtxt(args.tx_positions_file)
        # Ensure tx_positions is always 2D
        tx_positions = np.atleast_2d(tx_positions)
        print(f"Loaded {tx_positions.shape[0]} tx positions from {args.tx_positions_file}")

    num_positions = tx_positions.shape[0]

    # Process each tx_position and generate a corresponding csv file.
    for i, pos in enumerate(tx_positions):
        # Format the tx position as a comma-separated string with 2 decimals.
        tx_str = ",".join(f"{val:.2f}" for val in pos)
        csv_name = f"rss_{args.scene}_{tx_str}.csv"
        csv_path = os.path.join(args.out_dir, csv_name)
        
        # Generate the RSS CSV file for the current tx position.
        rss_array = rss_write_csv(scene_obj, tx_position=pos, csv_file=csv_path, compress=args.compress)
        print(f"[{i+1}/{num_positions}] RSS data for tx_position {tx_str} saved to {csv_path}")

if __name__ == '__main__':
    main()