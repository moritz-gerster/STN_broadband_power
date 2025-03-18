from os.path import join

import seaborn as sns

from scripts.config import FIG_PAPER
from scripts.plot_figures._correlation_by_frequencies import (
    get_corrs_kinds, plot_psd_updrs_correlation_multi)
from scripts.plot_figures._correlation_scatter_within import \
    normalized_beta_within
from scripts.plot_figures._correlation_within_by_bands import (
    barplot_biomarkers, get_correlation_df_multi)
from scripts.plot_figures._explain_repeated_measures import \
    repeated_measures_toy_example
from scripts.plot_figures._psd_clusters import (plot_psd_by_severity_conds,
                                                plot_psd_by_severity_kinds)
from scripts.plot_figures.settings import (BANDS, N_PERM_CLUSTER, get_dfs,
                                           sns_darkgrid)


def figure6(df_orig):
    # equalize subject count for model comparisons
    dataframes_equal = get_dfs(df_orig, ch_choice='ch_dist_sweet',
                               equalize_subjects_norm_abs=True)
    df_norm = dataframes_equal['df_norm']
    figure6a()
    figure6b()
    figure6c(df_norm)
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        figure6_def1(dataframes_equal)
        figure6_def2(dataframes_equal)
    figure6_def3(dataframes_equal)


def figure6a():
    """Plot toy example within vs across correlations."""
    repeated_measures_toy_example(fig_dir='Figure6', prefix='A__')


def figure6b():
    """Figure created externally."""
    pass


def figure6c(df_norm):
    """Relative beta does not correlate within subjects."""
    normalized_beta_within(df_norm, fig_dir='Figure6', prefix='C__')


def figure6_def1(dataframes):
    """Plot psd clusters within subjects all psd kinds."""
    kinds = ['absolute', 'periodic', 'periodicAP']
    conds = ['off', 'on']
    output_file_path = join(FIG_PAPER, 'Figure6', "D___output.txt")
    n_perm = N_PERM_CLUSTER
    with open(output_file_path, "w") as output_file:
        plot_psd_by_severity_kinds(dataframes, kinds, conds,
                                   lateralized_updrs=True,
                                   info_title=False, legend=True,
                                   figsize=(1.27, 1.2), xlabel=False,
                                   fig_dir='Figure6', prefix='D',
                                   within_comparison=True, stat_height=3e-4,
                                   output_file=output_file,
                                   n_perm=n_perm)
        plot_psd_by_severity_conds(dataframes, 'absolute', ['off'],
                                   lateralized_updrs=True,
                                   info_title=False,
                                   xscale='log', ylabel=False,
                                   within_comparison=True,
                                   legend=False, xlabel=False,
                                   yscale='log',
                                   xmin=13, xmax=200,
                                   xticks=[13, 50, 200],
                                   xticklabels=[13, 50, 200],
                                   yticks=[0.1, 1], yticklabels=[0.1, 1],
                                   ylim=(0.008, 1), stat_height=0.001,
                                   figsize=(.75, .6), n_perm=n_perm,
                                   fig_dir='Figure6', prefix='D7__',
                                   output_file=output_file)
        plot_psd_by_severity_conds(dataframes, 'absolute', ['on'],
                                   lateralized_updrs=True,
                                   info_title=False,
                                   xscale='log', ylabel=False,
                                   within_comparison=True,
                                   legend=False, xlabel=False,
                                   yscale='log',
                                   xmin=13, xmax=200,
                                   xticks=[13, 50, 200],
                                   xticklabels=[13, 50, 200],
                                   yticks=[0.1, 1], yticklabels=[0.1, 1],
                                   ylim=(0.006, 1), stat_height=0.0015,
                                   figsize=(.75, .6), n_perm=n_perm,
                                   fig_dir='Figure6', prefix='D8__',
                                   output_file=output_file)
        plot_psd_by_severity_conds(dataframes, 'periodic', ['off'],
                                   lateralized_updrs=True,
                                   info_title=False,
                                   xscale='log', ylabel=False,
                                   within_comparison=True,
                                   legend=False, xlabel=False,
                                   yscale='log',
                                   xmin=13, xmax=60,
                                   xticks=[13, 30, 60],
                                   xticklabels=[13, 30, 60],
                                   ylim=(4e-5, 1), stat_height=2e-5,
                                   figsize=(.8, .5), n_perm=n_perm,
                                   fig_dir='Figure6', prefix='D9__',
                                   output_file=output_file)
        plot_psd_by_severity_conds(dataframes, 'periodic', ['on'],
                                   lateralized_updrs=True,
                                   info_title=False,
                                   xscale='log', ylabel=False,
                                   within_comparison=True,
                                   legend=False, xlabel=False,
                                   yscale='log',
                                   xmin=13, xmax=60,
                                   xticks=[13, 30, 60],
                                   xticklabels=[13, 30, 60],
                                   ylim=(4e-5, 1), stat_height=2e-5,
                                   figsize=(.8, .5), n_perm=n_perm,
                                   fig_dir='Figure6', prefix='D10__',
                                   output_file=output_file)


def figure6_def2(dataframes):
    """PSD within-correlation by frequencies."""
    kinds = ['absolute', 'periodic', 'periodicAP']
    conds = ['off', 'on']
    df_corrs = get_corrs_kinds(dataframes, kinds, conds)
    plot_psd_updrs_correlation_multi(df_corrs, fig_dir='Figure6',
                                     prefix='E', legend=False, xmin=2,
                                     figsize=(1.27, 1),
                                     info_title=False,
                                     xlabel=None,
                                     ylim=(-0.22, 0.5))


def figure6_def3(dataframes):
    """Plot within-correlations by bands."""
    kinds = ['absolute', 'periodic', 'periodicAP']
    df_corr = get_correlation_df_multi(dataframes, kinds,
                                       corr_methods=['withinRank'],
                                       bands=BANDS+['gamma_mid'])
    output_file_path = join(FIG_PAPER, 'Figure6', "F___output.txt")
    with open(output_file_path, "w") as output_file:
        barplot_biomarkers(df_corr, fig_dir='Figure6', prefix='F',
                           height_stat=0.4, ylim=(-.4, .65),
                           figsize=(2.57, 1.3), output_file=output_file)