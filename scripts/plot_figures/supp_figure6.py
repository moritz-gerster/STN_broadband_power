from os.path import join

import seaborn as sns

from scripts import config as cfg
from scripts.plot_figures._correlation_by_frequencies import (
    get_corrs_kinds, plot_psd_updrs_correlation_multi)
from scripts.plot_figures._correlation_within_by_bands import (
    barplot_biomarkers, get_correlation_df_multi)
from scripts.plot_figures._hemisphere_comparison import \
    beta_peaks_by_hemisphere
from scripts.plot_figures._psd_clusters import plot_psd_by_severity_kinds
from scripts.plot_figures.settings import (BANDS, N_PERM_CLUSTER, get_dfs,
                                           sns_darkgrid)
from scripts.utils_plot import _dataset_overview


def supp_figure6(df_orig):
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df_per = dataframes['df_per']
    df_n = dataframes['df_sample_sizes']

    supp_figure6a(df_n)
    supp_figure6b(dataframes)
    supp_figure6c(df_per)


def supp_figure6a(df_n):
    _dataset_overview(df_n, fig_dir='Figure_S6', prefix='A__')


def supp_figure6b(dataframes):
    """Figure 7 for normalized data."""
    kinds = ['normalized']
    conds = ['off', 'on']
    fig_dir = 'Figure_S6'
    output_file_path = join(cfg.FIG_PAPER, fig_dir, "D1___output.txt")
    n_perm = N_PERM_CLUSTER
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        with open(output_file_path, "w") as output_file:
            plot_psd_by_severity_kinds(dataframes, kinds, conds,
                                       lateralized_updrs=True,
                                       info_title=None,
                                       legend=True,
                                       figsize=(1.27, 1.2), xlabel=False,
                                       ylim_norm=(0, 4), n_perm=n_perm,
                                       fig_dir=fig_dir, prefix="B",
                                       within_comparison=True,
                                       output_file=output_file)

        df_corrs = get_corrs_kinds(dataframes, kinds, conds)
        plot_psd_updrs_correlation_multi(df_corrs, fig_dir=fig_dir,
                                         prefix="B", legend=False, xmin=2,
                                         info_title=False,
                                         xlabel=None,
                                         ylim=(-0.3, 0.7),
                                         figsize=(1.27, 1))

    df_corr = get_correlation_df_multi(dataframes, kinds,
                                       corr_methods=['withinRank'],
                                       bands=BANDS+['gamma_mid'])
    output_file_path = join(cfg.FIG_PAPER, fig_dir, "D3___output.txt")
    with open(output_file_path, "w") as output_file:
        barplot_biomarkers(df_corr, fig_dir=fig_dir, prefix='B3__',
                           ylim=(-.3, .6), figsize=(2.57, 1.3),
                           output_file=output_file)


def supp_figure6c(df_per):
    """Count beta peaks."""
    beta_peaks_by_hemisphere(df_per, fig_dir='Figure_S6', prefix='C__')
