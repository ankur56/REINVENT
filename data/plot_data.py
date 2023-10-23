#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_data(load_dir, np_file, save_dir=None):
    """
    Plot data from a NumPy file and save the plot as a PNG.

    Parameters:
        load_dir (str): Directory to load the NumPy file from.
        np_file (str): Name of the NumPy file to plot.
        save_dir (str, optional): Directory to save the plot. If None, saves in current dir.
    """
    scores = np.load(os.path.join(load_dir, np_file))
    filename = os.path.splitext(os.path.basename(np_file))[0]

    plt.figure()
    plt.plot(scores, label=filename)
    plt.legend()
    plt.title(filename)
    plt.tight_layout()

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, filename)
    plt.savefig(filename + ".png", dpi=300)
    plt.close()

def main(args):
    load_dir = os.path.join('results', args.run_dir)
    save_dir = os.path.join('plots', args.run_dir)

    plot_data(load_dir, 'training_log_scores.npy', save_dir)
    plot_data(load_dir, 'training_log_valid.npy', save_dir)
    plot_data(load_dir, 'training_log_novel.npy', save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data from NumPy files.")
    parser.add_argument("run_dir", type=str, help="Directory name under 'results' containing NumPy files to plot.")
    args = parser.parse_args()

    main(args)



