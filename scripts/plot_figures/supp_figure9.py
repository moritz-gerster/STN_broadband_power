from os import makedirs
from os.path import join

import seaborn as sns

from scripts import config as cfg
from scripts.plot_figures._correlation_by_frequencies import (
    get_corrs_kinds, plot_psd_updrs_correlation_multi)
from scripts.plot_figures._correlation_within_by_bands import (
    barplot_biomarkers, get_correlation_df_multi)
from scripts.plot_figures._exemplary_subjects import exemplary_broadband_shifts
from scripts.plot_figures._psd_clusters import (plot_psd_by_severity_conds,
                                                plot_psd_by_severity_kinds)
from scripts.plot_figures.settings import (BANDS, N_PERM_CLUSTER, get_dfs,
                                           sns_darkgrid)


def supp_figure9(df_orig):
    # reproduction for adjacent channels
    dataframes_adjacent = get_dfs(df_orig,
                                  ch_choice='ch_adj_beta_high_max_off',
                                  equalize_subjects_norm_abs=True)
    df_per = dataframes_adjacent['df_per']
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        supp_figure9a(dataframes_adjacent)
        supp_figure9b(dataframes_adjacent)
    supp_figure9c(dataframes_adjacent)
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        supp_figure9d(df_per)


def supp_figure9a(dataframes):
    """Plot psd clusters within subjects all psd kinds."""
    kinds = ['absolute', 'periodic', 'periodicAP']
    conds = ['off', 'on']
    fig_dir = 'Figure_S9'
    output_file_path = join(cfg.FIG_PAPER, fig_dir, "A___output.txt")
    makedirs(join(cfg.FIG_PAPER, fig_dir), exist_ok=True)
    n_perm = N_PERM_CLUSTER
    with open(output_file_path, "w") as output_file:
        plot_psd_by_severity_kinds(dataframes, kinds, conds,
                                   lateralized_updrs=True,
                                   info_title=False, legend=True,
                                   figsize=(1.27, 1.2), xlabel=False,
                                   fig_dir=fig_dir, prefix='A',
                                   within_comparison=True, stat_height=1e-4,
                                   ylim_abs=(0.005, 5), ylim_ap=(0.005, 5),
                                   ylim_per=(-0.035, .5), n_perm=n_perm,
                                   output_file=output_file)
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
                                   figsize=(.7, .5), n_perm=n_perm,
                                   fig_dir=fig_dir, prefix='A7__',
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
                                   ylim=(0.004, 1), stat_height=0.0011,
                                   figsize=(.7, .5), n_perm=n_perm,
                                   fig_dir=fig_dir, prefix='A8__',
                                   output_file=output_file)
        plot_psd_by_severity_conds(dataframes, 'periodic', ['off'],
                                   lateralized_updrs=True,
                                   info_title=False,
                                   xscale='log', ylabel=False,
                                   within_comparison=True,
                                   legend=False, xlabel=False,
                                   yscale='log', n_perm=n_perm,
                                   xmin=13, xmax=60,
                                   xticks=[13, 30, 60],
                                   xticklabels=[13, 30, 60],
                                   ylim=(1e-6, .5), stat_height=1e-6,
                                   figsize=(.8, .5),
                                   fig_dir=fig_dir, prefix='A9__',
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
                                   ylim=(1e-8, .5), stat_height=2e-8,
                                   figsize=(.8, .5), n_perm=n_perm,
                                   fig_dir=fig_dir, prefix='A10__',
                                   output_file=output_file)


def supp_figure9b(dataframes):
    """PSD within-correlation by frequencies."""
    kinds = ['absolute', 'periodic', 'periodicAP']
    conds = ['off', 'on']
    fig_dir = 'Figure_S9'
    df_corrs = get_corrs_kinds(dataframes, kinds, conds)
    plot_psd_updrs_correlation_multi(df_corrs, fig_dir=fig_dir, prefix='B',
                                     legend=False, xmin=2, info_title=False,
                                     figsize=(1.27, 1),
                                     xlabel=None, ylim=(-0.3, 0.7))


def supp_figure9c(dataframes):
    """Plot within-correlations by bands."""
    kinds = ['absolute', 'periodic', 'periodicAP']
    df_corr = get_correlation_df_multi(dataframes, kinds,
                                       corr_methods=['withinRank'],
                                       bands=BANDS+['gamma_mid'])
    fig_dir = 'Figure_S9'
    output_file_path = join(cfg.FIG_PAPER, fig_dir, "C___output.txt")
    with open(output_file_path, "w") as output_file:
        barplot_biomarkers(df_corr, fig_dir=fig_dir, prefix='C__',
                           height_stat=0.6, fontsize_stat=7,
                           ylim=(-.4, .9), figsize=(2.57, 1.3),
                           output_file=output_file)


def supp_figure9d(df_per):
    """Exemplary periodic gamma oscillations."""
    exemplary_broadband_shifts(df_per, fig_dir='Figure_S9', prefix='D__',
                               subjects=cfg.EXEMPLARY_SUBS_GAMMA,
                               frameon=False,
                               annotate_gamma=True, aperiodic_shading=True)
