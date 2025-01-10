from os.path import join

import seaborn as sns

from scripts import config as cfg
from scripts.plot_figures._correlation_by_frequencies import (
    get_corrs_kinds, plot_psd_updrs_correlation_multi)
from scripts.plot_figures._correlation_scatter_within import \
    periodic_gamma_within
from scripts.plot_figures._correlation_within_by_bands import (
    barplot_biomarkers, get_correlation_df_multi)
from scripts.plot_figures._hemisphere_comparison import (
    beta_peaks_by_hemisphere, gamma_peaks_by_hemisphere,
    normalized_bands_by_hemisphere, periodic_bands_by_hemisphere)
from scripts.plot_figures._psd_clusters import plot_psd_by_severity_kinds
from scripts.plot_figures.settings import BANDS, get_dfs
from scripts.utils_plot import _dataset_overview


def supp_figure7(df_orig):
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df_per = dataframes['df_per']
    df_n = dataframes['df_sample_sizes']

    supp_figure7a(df_n)
    supp_figure7b(df_per)
    supp_figure7c(df_per)
    supp_figure7d(dataframes)

    # reproduction for max alpha-beta channels
    dataframes_abmax = get_dfs(df_orig,
                               ch_choice='ch_chmax_alpha_beta_abs_max_log')
    df_norm_adj = dataframes_abmax['df_norm']
    df_per_adj = dataframes_abmax['df_per']
    supp_figure7e(df_norm_adj)
    supp_figure7f(df_per_adj)


def supp_figure7a(df_n):
    _dataset_overview(df_n, fig_dir='Figure_S7', prefix='A__')


def supp_figure7b(df_per):
    """Periodic gamma does not correlate within subjects for distant LFP
    channels."""
    periodic_gamma_within(df_per, fig_dir='Figure_S7', prefix='B1__',
                          exemplary_subs=cfg.EXEMPLARY_SUBS_APERIODIC,
                          figsize=(1.7, 1.34))
    gamma_peaks_by_hemisphere(df_per, fig_dir='Figure_S7', prefix='B2__')


def supp_figure7c(df_per):
    """Count beta peaks."""
    beta_peaks_by_hemisphere(df_per, fig_dir='Figure_S7', prefix='C__')


def supp_figure7d(dataframes):
    """Figure 7 for nomalized data."""
    kinds = ['normalized']
    conds = ['off', 'on']
    fig_dir = 'Figure_S7'
    output_file_path = join(cfg.FIG_PAPER, fig_dir, "D1___output.txt")
    with sns.axes_style('darkgrid'):
        with open(output_file_path, "w") as output_file:
            plot_psd_by_severity_kinds(dataframes, kinds, conds,
                                       lateralized_updrs=True,
                                       info_title=None, legend=True,
                                       figsize=(2.4, 1.2), xlabel=False,
                                       ylim_norm=(0, 4),
                                       fig_dir=fig_dir, prefix="D1__",
                                       within_comparison=True,
                                       output_file=output_file)

        df_corrs = get_corrs_kinds(dataframes, kinds, conds)
        plot_psd_updrs_correlation_multi(df_corrs, fig_dir=fig_dir,
                                         prefix="D2__", legend=False, xmin=2,
                                         info_title=False,
                                         xlabel='Frequency (Hz)',
                                         ylim=(-0.3, 0.55), figsize=(2.4, 1.3))

    df_corr = get_correlation_df_multi(dataframes, kinds,
                                       corr_methods=['withinRank'],
                                       bands=BANDS+['gamma_mid'])
    output_file_path = join(cfg.FIG_PAPER, fig_dir, "D3___output.txt")
    with open(output_file_path, "w") as output_file:
        barplot_biomarkers(df_corr, fig_dir=fig_dir, prefix='D3__',
                           ylim=(-.3, .6), figsize=(2.4, 1.3),
                           output_file=output_file)


def supp_figure7e(df_norm_adj):
    """Reproduce Shreve et al."""
    normalized_bands_by_hemisphere(df_norm_adj, fig_dir='Figure_S7',
                                   prefix='E__')


def supp_figure7f(df_per_adj):
    """S7e for periodic power."""
    periodic_bands_by_hemisphere(df_per_adj, fig_dir='Figure_S7', prefix='F__')
