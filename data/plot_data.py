#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import trapz

def plot_data(load_dir, np_file, run_dir, save_dir=None):
    """
    Plot data from a NumPy file and save the plot as a PNG.

    Parameters:
        load_dir (str): Directory to load the NumPy file from.
        np_file (str): Name of the NumPy file to plot.
        save_dir (str, optional): Directory to save the plot. If None, saves in current dir.
    """
    scores = np.load(os.path.join(load_dir, np_file))
    filename = os.path.splitext(os.path.basename(np_file))[0]
    filename = run_dir + '_' + filename
    plt.figure()
    plt.plot(scores)
    #plt.legend()
    plt.title(filename, fontsize=6)
    if 'sigmas' not in np_file:
        plt.ylim(0, 1)
    plt.tight_layout()

    if 'st_scores' in np_file:
        x = np.arange(1, len(scores)+1)
        area = trapz(scores, x)
        plt.annotate(r'AUC = {:.2f}'.format(area), xy=(0.05, 0.95), xycoords='axes fraction')

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, filename)
    plt.savefig(filename + ".png", dpi=300)
    plt.close()

def main(args):
    load_dir = os.path.join('results', args.run_dir)
    save_dir = os.path.join('plots', args.run_dir)

    plot_data(load_dir, 'training_log_scores.npy', args.run_dir, save_dir)
    #plot_data(load_dir, 'training_log_sigmas.npy', args.run_dir, save_dir)
    plot_data(load_dir, 'training_log_st_scores.npy', args.run_dir, save_dir)
    plot_data(load_dir, 'training_log_valid.npy', args.run_dir, save_dir)
    plot_data(load_dir, 'training_log_novel.npy', args.run_dir, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data from NumPy files.")
    parser.add_argument("run_dir", type=str, help="Directory name under 'results' containing NumPy files to plot.")
    args = parser.parse_args()

    main(args)



