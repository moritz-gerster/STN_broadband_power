from os import makedirs
from os.path import join

import seaborn as sns

from scripts.config import FIG_PAPER
from scripts.plot_figures._correlation_by_frequencies import (
    df_corr_freq, plot_psd_updrs_correlation)
from scripts.plot_figures._exemplary_spectra import representative_spectrum
from scripts.plot_figures._forrest_plot_datasets_correlation import \
    forest_plot_correlation
from scripts.plot_figures._power_spectra import plot_normalized_spectra
from scripts.plot_figures._psd_clusters import plot_psd_by_severity_conds
from scripts.plot_figures.settings import (XTICKS_FREQ_low,
                                           XTICKS_FREQ_low_labels, get_dfs,
                                           sns_darkgrid)
from scripts.utils_plot import _dataset_density_plots, _dataset_histograms


def figure2(df_orig):
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df = dataframes['df']
    df_norm = dataframes['df_norm']
    figure2a(df)
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        figure2b(df_norm)
        figure2c(df_norm)
    figure2d(df_norm)
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        figure2e(dataframes)
        figure2f(df_norm)


def figure2a(df):
    """Dataset meta infos."""
    save_dir = join(FIG_PAPER, "Figure2")
    makedirs(save_dir, exist_ok=True)
    output_file_path = join(FIG_PAPER, 'Figure2', "A1___output.txt")
    with open(output_file_path, "w") as output_file:
        _dataset_histograms(df, save=True, save_dir=save_dir, prefix='A1__',
                            output_file=output_file)
    _dataset_density_plots(df, save=True, save_dir=save_dir, prefix='A2__')


def figure2b(df_norm):
    """Exemplary spectrum normalized."""
    output_file_path = join(FIG_PAPER, 'Figure2', "B___output.txt")
    with open(output_file_path, "w") as output_file:
        representative_spectrum(df_norm, 'normalized',
                                fig_dir='Figure2',
                                legend=True, yscale='linear',
                                figsize=(1.7, 1.65), xlabel=False,
                                output_file=output_file,
                                prefix='B1__')
        representative_spectrum(df_norm, 'normalized',
                                fig_dir='Figure2',
                                legend=False,
                                figsize=(1.65, 1.6895), xlabel=False,
                                xscale='log', ylabel='',
                                output_file=output_file,
                                prefix='B2__')


def figure2c(df_norm):
    """Normalized power spectra by dataset."""
    plot_normalized_spectra(df_norm, fig_dir='Figure2', prefix='C__')


def figure2d(df_norm):
    """Beta ~ UPDRS-III correlation by dataset."""
    cond = 'off'
    df_cond = df_norm[df_norm.cond == cond]
    bands = ['alpha_beta_abs_mean_log',
             'beta_abs_mean_log',
             'beta_low_abs_mean_log']
    output_file_path = join(FIG_PAPER, 'Figure2', "D___output.txt")
    with open(output_file_path, "w") as output_file:
        forest_plot_correlation(df_cond, bands, 'UPDRS_III',
                                fig_dir='Figure2', prefix='D__',
                                dataset_labels=True,
                                figsize=(3.2, 1.2), markerscale=0.1,
                                y_fontsize=6.5, title_y=False,
                                xlabel=False, output_file=output_file)


def figure2e(dataframes):
    """PSDs by severity conditions."""
    output_file_path = join(FIG_PAPER, 'Figure2', "E___output.txt")

    with open(output_file_path, "w") as output_file:
        plot_psd_by_severity_conds(dataframes, 'normalized', ['off'],
                                   lateralized_updrs=False,
                                   color_by_kind=False,
                                   xmin=2, xmax=45, info_title=False,
                                   figsize=(1.9, 1.3295), fig_dir='Figure2',
                                   xticks=XTICKS_FREQ_low,
                                   xticklabels=XTICKS_FREQ_low_labels,
                                   ylim=(0, 8), prefix='E__', xlabel=False,
                                   output_file=output_file)


def figure2f(df_norm):
    """Correlation by frequency bin."""
    x = 'psd_log'
    y = 'UPDRS_III'
    corr_method = 'spearman'
    data = df_norm[(df_norm.cond == 'off')]
    df_freqs = df_corr_freq(data, x, y, corr_method=corr_method)

    output_file_path = join(FIG_PAPER, 'Figure2', "F___output.txt")
    with open(output_file_path, "w") as output_file:
        plot_psd_updrs_correlation(df_freqs, x, y, 'normalized',
                                   fig_dir='Figure2', prefix='F__',
                                   xlabel=False, output_file=output_file)