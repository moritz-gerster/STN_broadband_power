from os.path import join

import matplotlib.pyplot as plt
import numpy as np

import scripts.config as cfg
from scripts.plot_figures.settings import XTICKS_FREQ_low
from scripts.utils_plot import _save_fig


def plot_beta_ranges(fig_dir='Figure3', prefix='', xlabel=True):
    frequency_ranges_all = [(8, 35), (8, 35), (8, 35), (8, 35), (13, 35),
                            (13, 35), (12, 20), (8, 35), (13, 35), (12, 33),
                            (13, 30), (13, 30), (8, 35), (13, 20), (21, 30),
                            (13, 20), (13, 30), (10, 14), (8, 35), (13, 35),
                            (8, 35), (8, 35), (13, 30), (13, 30), (13, 30),
                            (13, 22), (12, 30), (12, 30), (13, 35), (12, 20),
                            (13, 20), (13, 20), (13, 35), (13, 35), (13, 35),
                            (13, 35), (13, 35), (13, 30), (13, 30)]

    alphabeta = cfg.BAND_NAMES_GREEK_SHORT['alpha_beta']
    beta = cfg.BAND_NAMES_GREEK_SHORT['beta']
    lowbeta = cfg.BAND_NAMES_GREEK_SHORT['beta_low']

    band_ranges = [(13, 20), (13, 30), (8, 35)]
    sorted_ranges_all = [(13, 20), (13, 30), (8, 35)] + frequency_ranges_all
    ypositions = np.arange(1, len(sorted_ranges_all))

    # Create a figure and axes
    height = 1.25 if xlabel else 1.1
    fig, axes = plt.subplots(2, 1, figsize=(1.2, height), height_ratios=[4, 1])

    ax = axes[0]
    for i, freq_range in enumerate(frequency_ranges_all):
        ax.hlines(ypositions[i], freq_range[0], freq_range[1], colors='grey')

    # Set labels and title
    ax.set_ylabel('Study index (table S1)')
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    ax.set_yticklabels([1, '', 10, '', 20, '', 30, '', 40, ''])
    ax.set_ylim(0, len(sorted_ranges_all)-2)
    ax.set_xticks(XTICKS_FREQ_low[1:], labels=[])

    ax = axes[1]
    for i, freq_range in enumerate(band_ranges):
        ax.hlines(i, freq_range[0], freq_range[1], colors='k')
    ax.set_yticks(range(3))
    ax.set_yticklabels([lowbeta, beta, alphabeta])

    xlabel = f'{cfg.BAND_NAMES_GREEK['beta']} ranges [Hz]' if xlabel else None
    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_xticks(XTICKS_FREQ_low[1:])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    save_dir = join(cfg.FIG_PAPER, fig_dir)
    fig_name = f'{prefix}study_beta_ranges_off.pdf'
    _save_fig(fig, fig_name, save_dir, bbox_inches=None, close=False)
