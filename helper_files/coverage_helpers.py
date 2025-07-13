import numpy as np
import pandas as pd
import tensorflow as tf
from scene_helpers import remove_all_transmitters, grid_indices_to_center_coordinate
import re
import os

MAX_DEPTH=30
CELL_SIZE=(2,2,2)  # Use a tuple of length 2 for Sionna RT compatibility
SAMPLES_PER_TX = 10**8
THRESHOLD = -100  # dBm threshold for coverage calculation

# Import relevant components from Sionna RT
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies


def rss_map_full(scene, tx_position=[8.5,21,27], max_depth=MAX_DEPTH, cell_size=CELL_SIZE, samples_per_tx=SAMPLES_PER_TX, csv_file="full_rss_map.csv"):
    remove_all_transmitters(scene)
    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="V")

    # Create transmitter
    tx = Transmitter(name="tx",
                    position= tx_position,
                    display_radius=2)

    # Add transmitter instance to scene
    scene.add(tx)
    rm_solver = RadioMapSolver()
    # Ensure cell_size is a tuple of length 2 for Sionna RT
    cell_size_2d = tuple(cell_size[:2])
    rm = rm_solver(scene=scene,
                   max_depth=max_depth,
                   cell_size=cell_size_2d,
                   samples_per_tx=samples_per_tx)
    return rm.rss.numpy() if hasattr(rm.rss, 'numpy') else rm.rss

def compute_coverage(scene, tx_position=[8.5,21,27]):
    rss = rss_map_full(scene, tx_position=tx_position, csv_file=None)

    # Configure antenna array for all transmitters    
    # Convert rss tensor to numpy array if necessary
    if hasattr(rss, 'numpy'):
        rss_array = rss.numpy()
    else:
        rss_array = np.array(rss)
            
    coverage = compute_coverage_from_arr(rss_array, threshold_dbm=-100)
    return coverage


def compute_coverage_from_arr(rss_array, threshold_dbm=-100):
    """
    Compute coverage from RSS array.
    Coverage is defined as the ratio of the area where RSS is above the threshold.
    
    Args:
        rss_array (numpy.ndarray): 2D array of RSS values in absolute (linear) scale
        threshold_dbm (float): RSS threshold for coverage calculation in dBm (default: -100 dBm)
    
    Returns:
        float: Coverage value (ratio of area above threshold)
    """
    # Convert absolute RSS values to dBm
    # dBm = 10 * log10(absolute_value)
    with np.errstate(divide='ignore'):  # Ignore warnings about log of zero
        rss_dbm = 10 * np.log10(rss_array)
    
    # Count total and above-threshold samples
    total_samples = rss_array.size
    above_threshold_samples = np.sum(rss_dbm > threshold_dbm)
    
    # Compute coverage as ratio
    coverage = above_threshold_samples / total_samples
    
    return coverage

def compute_coverage_from_csv(csv_file):
    """
    Computes the coverage from a CSV file containing RSS data.
    
    Args:
        csv_file (str): Path to the CSV file.
        
    Returns:
        float: Coverage value.
    """
    df = pd.read_csv(csv_file)
    rss_array = df.values
    threshold = THRESHOLD
    ones_count = np.sum(rss_array > threshold)
    total_values = rss_array.size
    
    coverage = ones_count / total_values
    return coverage

def compute_coverage_from_csv_gz(file_path, threshold_dbm=THRESHOLD, scene_name="munich"):
    """
    Given a .csv.gz RSS file whose name encodes the transmitter coordinates, compute the coverage.
    Args:
        file_path (str): Path to the .csv.gz file (e.g., rss_munich_625.42,-457.60,20.00.csv.gz)
        threshold_dbm (float): Coverage threshold in dBm (default: global THRESHOLD)
        scene_name (str): Scene name (default: "munich")
    Returns:
        tuple: (tx_x, tx_y, tx_z, coverage)
    """
    # Extract coordinates from filename
    filename = os.path.basename(file_path)
    m = re.match(r"rss_" + re.escape(scene_name) + r"_([\-\d.]+),([\-\d.]+),([\-\d.]+)\.csv\.gz", filename)
    if not m:
        raise ValueError(f"Filename {filename} does not match expected pattern for scene '{scene_name}'.")
    tx_x, tx_y, tx_z = map(float, m.groups())
    # Read RSS array
    rss_array = pd.read_csv(file_path, header=None).values
    coverage = compute_coverage_from_arr(rss_array, threshold_dbm=threshold_dbm)
    return tx_x, tx_y, tx_z, coverage


def compute_coverage_for_directory_to_csv(dir_path, output_csv, threshold_dbm=THRESHOLD, scene_name="munich"):
    """
    For all rss_<scene_name>_*.csv.gz files in the directory, compute coverage and write to a CSV file.
    Each line in the output CSV will be: x, y, z, coverage
    Args:
        dir_path (str): Directory containing rss_<scene_name>_*.csv.gz files
        output_csv (str): Path to output CSV file
        threshold_dbm (float): Coverage threshold in dBm (default: global THRESHOLD)
        scene_name (str): Scene name (default: "munich")
    """
    import glob
    results = []
    pattern = os.path.join(dir_path, f'rss_{scene_name}_*.csv.gz')
    files = glob.glob(pattern)
    total = len(files)
    print(f"Found {total} files to process.")
    for idx, file_path in enumerate(files, 1):
        try:
            tx_x, tx_y, tx_z, coverage = compute_coverage_from_csv_gz(file_path, threshold_dbm=threshold_dbm, scene_name=scene_name)
            results.append([tx_x, tx_y, tx_z, coverage])
            print(f"\rProcessed {idx}/{total}: {os.path.basename(file_path)}", end="", flush=True)
        except Exception as e:
            print(f"\nSkipping {file_path}: {e}")
    print("\nAll files processed.")
    df = pd.DataFrame(results, columns=["x", "y", "z", "coverage"])
    df.to_csv(output_csv, index=False)
    print(f"Wrote coverage data for {len(results)} files to {output_csv}")

def find_closest_rss_file(coord, rm_data_dir, scene_name="munich"): 
    """
    Given a coordinate (x, y, z) and a directory containing rss_<scene_name>_<x>,<y>,<z>.csv.gz files,
    return the path to the file whose name has the closest coordinate.
    Args:
        coord (tuple/list): (x, y, z) coordinate
        rm_data_dir (str): Path to directory containing rss files
        scene_name (str): Scene name (default: "munich")
    Returns:
        str: Path to the closest file
    """
    import os
    import re
    import numpy as np
    pattern = re.compile(rf"rss_{scene_name}_([\-\d.]+),([\-\d.]+),([\-\d.]+)\.csv\.gz$")
    min_dist = float('inf')
    closest_file = None
    for fname in os.listdir(rm_data_dir):
        m = pattern.match(fname)
        if m:
            file_coord = np.array(list(map(float, m.groups())))
            dist = np.linalg.norm(np.array(coord) - file_coord)
            if dist < min_dist:
                min_dist = dist
                closest_file = os.path.join(rm_data_dir, fname)
    return closest_file

def compute_coverage_for_closest_coordinate(coord, rm_data_dir, threshold_dbm=THRESHOLD, scene_name="munich"):
    """
    Given a coordinate and a directory of rss_<scene_name>_<x>,<y>,<z>.csv.gz files,
    find the file with the closest coordinate and compute its coverage.
    Args:
        coord (tuple/list): (x, y, z) coordinate
        rm_data_dir (str): Path to directory containing rss files
        threshold_dbm (float): Coverage threshold in dBm (default: global THRESHOLD)
        scene_name (str): Scene name (default: "munich")
    Returns:
        tuple: (closest_coord, coverage, file_path)
    """
    file_path = find_closest_rss_file(coord, rm_data_dir, scene_name=scene_name)
    if file_path is None:
        raise FileNotFoundError("No matching RSS file found in directory.")
    # Extract coordinate from filename
    import re, os
    fname = os.path.basename(file_path)
    m = re.match(rf"rss_{scene_name}_([\-\d.]+),([\-\d.]+),([\-\d.]+)\.csv\.gz$", fname)
    if not m:
        raise ValueError(f"Filename {fname} does not match expected pattern for scene '{scene_name}'.")
    closest_coord = tuple(map(float, m.groups()))
    #print(f"Closest coordinate found: {closest_coord} in file {file_path}")
    coverage = compute_coverage_from_csv_gz(file_path, threshold_dbm=threshold_dbm)
    return coverage[-1] #closest_coord, coverage, file_path