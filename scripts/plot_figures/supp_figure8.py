from os import makedirs
from os.path import join

import seaborn as sns

from scripts import config as cfg
from scripts.plot_figures._correlation_by_frequencies import (
    get_corrs_kinds, plot_psd_updrs_correlation_multi)
from scripts.plot_figures._correlation_scatter_within import \
    periodic_gamma_within
from scripts.plot_figures._correlation_within_by_bands import (
    barplot_biomarkers, get_correlation_df_multi)
from scripts.plot_figures._exemplary_subjects import exemplary_gamma
from scripts.plot_figures._psd_clusters import (plot_psd_by_severity_conds,
                                                plot_psd_by_severity_kinds)
from scripts.plot_figures.settings import BANDS, get_dfs


def supp_figure8(df_orig):
    # reproduction for adjacent channels
    dataframes_adjacent = get_dfs(df_orig,
                                  ch_choice='ch_adj_beta_high_max_off',
                                  equalize_subjects_norm_abs=True)
    df_per = dataframes_adjacent['df_per']
    with sns.axes_style('darkgrid'):
        supp_figure8a(dataframes_adjacent)
        supp_figure8b(dataframes_adjacent)
    supp_figure8c(dataframes_adjacent)
    supp_figure8d(df_per)
    with sns.axes_style('darkgrid'):
        supp_figure8e(df_per)


def supp_figure8a(dataframes):
    """Plot psd clusters within subjects all psd kinds."""
    kinds = ['absolute', 'periodic', 'periodicAP']
    conds = ['off', 'on']
    fig_dir = 'Figure_S8'
    output_file_path = join(cfg.FIG_PAPER, fig_dir, "A___output.txt")
    makedirs(join(cfg.FIG_PAPER, fig_dir), exist_ok=True)
    with open(output_file_path, "w") as output_file:
        plot_psd_by_severity_kinds(dataframes, kinds, conds,
                                   lateralized_updrs=True,
                                   info_title=None, legend=True,
                                   figsize=(7, 1.2), xlabel=False,
                                   fig_dir=fig_dir, prefix='A1__',
                                   within_comparison=True, stat_height=3e-4,
                                   ylim_abs=(0.005, 5), ylim_ap=(0.005, 5),
                                   ylim_per=(-0.035, .5),
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
                                   figsize=(.7, .5),
                                   fig_dir=fig_dir, prefix='AX__',
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
                                   figsize=(.7, .5),
                                   fig_dir=fig_dir, prefix='A2__',
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
                                   ylim=(1e-6, .5), stat_height=1e-6,
                                   figsize=(.8, .5),
                                   fig_dir=fig_dir, prefix='A3__',
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
                                   ylim=(1e-8, .5), stat_height=1e-8,
                                   figsize=(.8, .5),
                                   fig_dir=fig_dir, prefix='A4__',
                                   output_file=output_file)


def supp_figure8b(dataframes):
    """PSD within-correlation by frequencies."""
    kinds = ['absolute', 'periodic', 'periodicAP']
    conds = ['off', 'on']
    fig_dir = 'Figure_S8'
    df_corrs = get_corrs_kinds(dataframes, kinds, conds)
    plot_psd_updrs_correlation_multi(df_corrs, fig_dir=fig_dir, prefix='B__',
                                     legend=False, xmin=2, info_title=False,
                                     xlabel='Frequency (Hz)', ylim=(-0.3, 0.7))


def supp_figure8c(dataframes):
    """Plot within-correlations by bands."""
    kinds = ['absolute', 'periodic', 'periodicAP']
    df_corr = get_correlation_df_multi(dataframes, kinds,
                                       corr_methods=['withinRank'],
                                       bands=BANDS+['gamma_mid'])
    fig_dir = 'Figure_S8'
    output_file_path = join(cfg.FIG_PAPER, fig_dir, "C___output.txt")
    with open(output_file_path, "w") as output_file:
        barplot_biomarkers(df_corr, fig_dir=fig_dir, prefix='C__',
                           ylim=(-.3, .7), figsize=(7, 1.3),
                           output_file=output_file)


def supp_figure8d(df_per):
    """Periodic gamma does correlate within subjects for adjacent LFP
    channels."""
    exemplary_subs = cfg.EXEMPLARY_SUBS_GAMMA
    periodic_gamma_within(df_per, fig_dir='Figure_S8', prefix='D__',
                          exemplary_subs=exemplary_subs, figsize=(1.55, 1.34),
                          bbox_to_anchor=(-.3, 1))


def supp_figure8e(df_per):
    """Exemplary periodic gamma oscillations."""
    exemplary_gamma(df_per, fig_dir='Figure_S8', prefix='E__')
