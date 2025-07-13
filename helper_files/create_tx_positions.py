# Other imports
import matplotlib.pyplot as plt

from tx_position import find_best_tx_position_rand, find_best_tx_position_bayesian
from coverage_helpers import *
from scene_helpers import *

import subprocess
import numpy as np
import os
import argparse


N_INITIAL_POS = 5
N_ITERATIONS = 5

# Example usage
scene = get_sionna_scene()

# Search space 
x_min, x_max, y_min, y_max, z_min, z_max = get_scene_bounds3d(scene)
bounds = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])

def main():
    parser = argparse.ArgumentParser(description="Generate grid of transmitter positions for radio map data.")
    parser.add_argument('--step_size', type=int, default=100, help='Grid step size (default: 100)')
    parser.add_argument('--scene', type=str, default='munich', help='Scene name (default: "munich")')
    parser.add_argument('--z_height', type=float, default=20.0, help='Z height for all positions (default: 20.0)')
    args = parser.parse_args()

    # Load scene
    scene = get_sionna_scene(getattr(__import__('sionna.rt.scene', fromlist=[args.scene]), args.scene))
    x_min, x_max, y_min, y_max, z_min, z_max = get_scene_bounds3d(scene)
    step_size = args.step_size
    num_offsets = step_size
    z_height = args.z_height

    all_positions = []
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    max_offsets_x = int(x_range / CELL_SIZE[0])
    max_offsets_y = int(y_range / CELL_SIZE[1])
    max_offsets_z = int(z_range / CELL_SIZE[2])

    print(f"Maximum offsets: X={max_offsets_x}, Y={max_offsets_y}, Z={max_offsets_z}")

    positions_file = f'rm_data/tx_positions_grid_{args.scene}_step{step_size}.csv'
    os.makedirs('rm_data', exist_ok=True)

    for offset in range(num_offsets):
        x_centers = np.arange(x_min + offset, x_max, step_size)
        y_centers = np.arange(y_min + offset, y_max, step_size)
        X, Y = np.meshgrid(x_centers, y_centers)
        positions = np.column_stack((X.flatten(), Y.flatten(), np.full(X.size, z_height)))
        all_positions.append(positions)
        print(f'Grid {offset}: Generated {len(positions)} positions with offset {offset}')

    positions = np.vstack(all_positions)
    np.savetxt(positions_file, positions, delimiter=',')
    print(f'\nTotal positions generated: {len(positions)}')
    print(f'Saved all positions to {positions_file}')

    print(f"\nSummary:")
    print(f"  Number of grids (offsets): {num_offsets}")
    print(f"  Positions per grid (approx): {(positions.shape[0] // num_offsets) if num_offsets else positions.shape[0]}")
    print(f"  Total unique positions: {len(positions)}")
    print(f"  Output file: {positions_file}")

if __name__ == '__main__':
    main()