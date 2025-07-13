#!/usr/bin/env python3
# filepath: /home/neel/gpsharma/Capstone-Project-COVER/read_rss_csv.py
import argparse
import pandas as pd
import numpy as np
import os
import re

def read_csv_to_numpy(csv_file):
    """
    Reads a CSV file and returns its data as a NumPy array.
    
    Args:
        csv_file (str): Path to the CSV file.
        
    Returns:
        numpy.ndarray: The array containing CSV data.
    """
    df = pd.read_csv(csv_file, header=None)
    return df.values

def get_available_tx_coordinates_from_dir(dir):
    """
    Scans the given directory for files named like 'rss_munich_<x>,<y>,<z>.csv.gz' and returns an array of transmitter coordinates (x, y, z) for which coverage files are available.
    
    Args:
        dir (str): Path to the dataset directory.
        
    Returns:
        np.ndarray: Array of shape (N, 3) with transmitter coordinates for available coverage files.
    """
    coords = []
    pattern = re.compile(r"rss_munich_([-\d.]+),([-\d.]+),([-\d.]+)\.csv\.gz$")
    for fname in os.listdir(dir):
        match = pattern.match(fname)
        if match:
            x, y, z = map(float, match.groups())
            coords.append([x, y, z])
    return np.array(coords)



def main():
    parser = argparse.ArgumentParser(description="Read an RSS CSV file into a NumPy array.")
    parser.add_argument("csv_file", type=str, help="Path to the RSS CSV file.")
    args = parser.parse_args()

    data_array = read_csv_to_numpy(args.csv_file)
    print("Loaded array shape:", data_array.shape)
if __name__ == "__main__":
    main()