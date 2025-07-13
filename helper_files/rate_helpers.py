import numpy as np
import pandas as pd
import os
import re
from scene_helpers import grid_indices_to_center_coordinate

NOISE_POWER = 1e-10  # Example: -100 dBm ≈ 1e-10 W (set as needed)

def compute_rate_from_arr(rss_array, noise_power=NOISE_POWER):
    """
    Compute average rate from RSS array using Shannon capacity formula.
    Args:
        rss_array (numpy.ndarray): 2D array of RSS values in absolute (linear) scale (Watts)
        noise_power (float): Noise power in Watts
    Returns:
        float: Average rate (bits/s/Hz) over the grid
    """
    with np.errstate(divide='ignore'):
        snr = rss_array / noise_power
        rate = np.log2(1 + snr)
    avg_rate = np.mean(rate)
    return avg_rate

def compute_rate_from_csv(csv_file, noise_power=NOISE_POWER):
    """
    Computes the average rate from a CSV file containing RSS data.
    Args:
        csv_file (str): Path to the CSV file.
        noise_power (float): Noise power in Watts
    Returns:
        float: Average rate (bits/s/Hz)
    """
    df = pd.read_csv(csv_file, header=None)
    rss_array = df.values
    return compute_rate_from_arr(rss_array, noise_power=noise_power)

def compute_rate_from_csv_gz(file_path, noise_power=NOISE_POWER, scene_name="munich"):
    """
    Given a .csv.gz RSS file whose name encodes the transmitter coordinates, compute the average rate.
    Args:
        file_path (str): Path to the .csv.gz file (e.g., rss_munich_625.42,-457.60,20.00.csv.gz)
        noise_power (float): Noise power in Watts
        scene_name (str): Scene name (default: "munich")
    Returns:
        tuple: (tx_x, tx_y, tx_z, avg_rate)
    """
    filename = os.path.basename(file_path)
    m = re.match(r"rss_" + re.escape(scene_name) + r"_([\-\d.]+),([\-\d.]+),([\-\d.]+)\.csv\.gz", filename)
    if not m:
        raise ValueError(f"Filename {filename} does not match expected pattern for scene '{scene_name}'.")
    tx_x, tx_y, tx_z = map(float, m.groups())
    rss_array = pd.read_csv(file_path, header=None).values
    avg_rate = compute_rate_from_arr(rss_array, noise_power=noise_power)
    return tx_x, tx_y, tx_z, avg_rate

def compute_rate_for_directory_to_csv(dir_path, output_csv, noise_power=NOISE_POWER, scene_name="munich"):
    """
    For all rss_<scene_name>_*.csv.gz files in the directory, compute average rate and write to a CSV file.
    Each line in the output CSV will be: x, y, z, avg_rate
    Args:
        dir_path (str): Directory containing rss_<scene_name>_*.csv.gz files
        output_csv (str): Path to output CSV file
        noise_power (float): Noise power in Watts
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
            tx_x, tx_y, tx_z, avg_rate = compute_rate_from_csv_gz(file_path, noise_power=noise_power, scene_name=scene_name)
            results.append([tx_x, tx_y, tx_z, avg_rate])
            print(f"\rProcessed {idx}/{total}: {os.path.basename(file_path)}", end="", flush=True)
        except Exception as e:
            print(f"\nSkipping {file_path}: {e}")
    print("\nAll files processed.")
    df = pd.DataFrame(results, columns=["x", "y", "z", "avg_rate"])
    df.to_csv(output_csv, index=False)
    print(f"Wrote rate data for {len(results)} files to {output_csv}")

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

def compute_rate_for_closest_coordinate(coord, rm_data_dir, noise_power=NOISE_POWER, scene_name="munich"):
    """
    Given a coordinate and a directory of rss_<scene_name>_<x>,<y>,<z>.csv.gz files,
    find the file with the closest coordinate and compute its average rate.
    Args:
        coord (tuple/list): (x, y, z) coordinate
        rm_data_dir (str): Path to directory containing rss files
        noise_power (float): Noise power in Watts
        scene_name (str): Scene name (default: "munich")
    Returns:
        float: avg_rate at the closest coordinate
    """
    file_path = find_closest_rss_file(coord, rm_data_dir, scene_name=scene_name)
    if file_path is None:
        raise FileNotFoundError("No matching RSS file found in directory.")
    avg_rate = compute_rate_from_csv_gz(file_path, noise_power=noise_power, scene_name=scene_name)[-1]
    return avg_rate