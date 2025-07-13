import argparse
import os
import numpy as np
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Generate RSS CSV files only for missing positions.")
    parser.add_argument('--tx_positions_file', type=str, required=True, help='CSV file with transmitter positions (x,y,z per row)')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory where RSS files are/will be stored (e.g., ./rm_data/)')
    parser.add_argument('--scene', type=str, default='munich', help='Scene name (default: "munich")')
    parser.add_argument('--compress', action='store_true', help='Compress output files using gzip')
    args = parser.parse_args()

    # Load all positions
    positions = np.loadtxt(args.tx_positions_file, delimiter=',')
    positions = np.atleast_2d(positions)

    # List all existing files in dataset_dir
    existing_files = set(os.listdir(args.dataset_dir))
    remaining_positions = []

    for pos in positions:
        x, y, z = pos
        expected_file = f"rss_{args.scene}_{x:.2f},{y:.2f},{z:.2f}.csv.gz" if args.compress else f"rss_{args.scene}_{x:.2f},{y:.2f},{z:.2f}.csv"
        if expected_file not in existing_files:
            remaining_positions.append(pos)

    if not remaining_positions:
        print("All positions already have RSS files. Nothing to do.")
        return

    print(f"{len(remaining_positions)} positions to process out of {len(positions)} total.")
    temp_file = os.path.join(args.dataset_dir, 'remaining_tx_positions.csv')
    np.savetxt(temp_file, remaining_positions, delimiter=',')

    cmd = [
        'python', 'generate_rss_csv.py',
        '--scene', args.scene,
        '--tx_positions_file', temp_file,
        '--out_dir', args.dataset_dir
    ]
    if args.compress:
        cmd.append('--compress')
    print('Running command:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
