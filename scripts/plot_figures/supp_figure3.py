from os.path import join

import seaborn as sns

from scripts.config import FIG_PAPER
from scripts.plot_figures._correlation_by_bands import barplot_UPDRS_bands
from scripts.plot_figures._correlation_by_frequencies import (
    df_corr_freq, plot_psd_updrs_correlation)
from scripts.plot_figures._forrest_plot_datasets_correlation import \
    forest_plot_correlation
from scripts.plot_figures._psd_clusters import plot_psd_by_severity_conds
from scripts.plot_figures.settings import (BANDS, N_PERM_CORR, XTICKS_FREQ_low,
                                           XTICKS_FREQ_low_labels, get_dfs,
                                           sns_darkgrid)
from scripts.utils import get_correlation_df


def supp_figure3(df_orig):
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df_norm = dataframes['df_norm']
    df_norm = df_norm[df_norm.cond == 'offon_abs']
    supp_figure3a(df_norm)
    supp_figure3b(df_norm)
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        supp_figure3c(dataframes)
        supp_figure3d(df_norm)
    supp_figure3e(df_norm)
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        supp_figure3f(dataframes)
        supp_figure3g(df_norm)
    supp_figure3h(df_norm)


def supp_figure3a(df_norm):
    """Beta ~ UPDRS-III correlation by dataset."""
    bands = ['alpha_beta_abs_mean_log',
             'beta_abs_mean_log',
             'beta_low_abs_mean_log']
    forest_plot_correlation(df_norm, bands, 'UPDRS_III', fig_dir='Figure_S3',
                            prefix='A__', dataset_labels=True)


def supp_figure3b(df_norm):
    """Beta ~ Bradykinesia-Rigdity correlation by dataset."""
    bands = ['alpha_beta_abs_mean_log',
             'beta_abs_mean_log',
             'beta_low_abs_mean_log']
    forest_plot_correlation(df_norm, bands, 'UPDRS_bradyrigid_contra',
                            fig_dir='Figure_S3', prefix='B__',
                            dataset_labels=False)


def supp_figure3c(dataframes):
    """PSDs by severity conditions."""
    output_file_path = join(FIG_PAPER, 'Figure_S3', "C___output.txt")

    with open(output_file_path, "w") as output_file:
        plot_psd_by_severity_conds(dataframes, 'normalized', ['offon_abs'],
                                   lateralized_updrs=False,
                                   color_by_kind=False,
                                   xmin=2, xmax=45, info_title=False,
                                   figsize=(1.9, 1.3), fig_dir='Figure_S3',
                                   xticks=XTICKS_FREQ_low,
                                   xticklabels=XTICKS_FREQ_low_labels,
                                   ylim=(0, 8), prefix='C__',
                                   output_file=output_file)


def supp_figure3f(dataframes):
    """PSDs by severity conditions."""
    output_file_path = join(FIG_PAPER, 'Figure_S3', "F___output.txt")

    with open(output_file_path, "w") as output_file:
        plot_psd_by_severity_conds(dataframes, 'normalized', ['offon_abs'],
                                   lateralized_updrs=True, color_by_kind=False,
                                   xmin=2, xmax=45, info_title=False,
                                   figsize=(1.9, 1.3), fig_dir='Figure_S3',
                                   xticks=XTICKS_FREQ_low,
                                   xticklabels=XTICKS_FREQ_low_labels,
                                   ylim=(0, 8), prefix='F__',
                                   output_file=output_file)


def supp_figure3d(df_norm):
    """Correlation by frequency bin."""
    x = 'psd_log'
    y = 'UPDRS_III'
    df_freqs = df_corr_freq(df_norm, x, y)

    output_file_path = join(FIG_PAPER, 'Figure_S3', "D___output.txt")
    with open(output_file_path, "w") as output_file:
        plot_psd_updrs_correlation(df_freqs, x, y, 'normalized',
                                   fig_dir='Figure_S3', prefix='D__',
                                   output_file=output_file)


def supp_figure3g(df_norm):
    """Correlation by frequency bin."""
    x = 'psd_log'
    y = 'UPDRS_bradyrigid_contra'
    df_freqs = df_corr_freq(df_norm, x, y)

    output_file_path = join(FIG_PAPER, 'Figure_S3', "G___output.txt")
    with open(output_file_path, "w") as output_file:
        plot_psd_updrs_correlation(df_freqs, x, y, 'normalized',
                                   fig_dir='Figure_S3', prefix='G__',
                                   output_file=output_file)


def supp_figure3e(df_norm):
    """Plot correlations by bands."""
    y = 'UPDRS_III'
    output_file_path = join(FIG_PAPER, 'Figure_S3', "E___output.txt")
    with open(output_file_path, "w") as output_file:
        df_corr = get_correlation_df(df_norm, y,
                                     n_perm=N_PERM_CORR,
                                     use_peak_power=True, bands=BANDS,
                                     output_file=output_file)
    df_corr['kind'] = 'normalized'
    barplot_UPDRS_bands(df_corr, fig_dir='Figure_S3', prefix='E__',
                        fontsize_stat=8)


def supp_figure3h(df_norm):
    """Plot correlations by bands."""
    y = 'UPDRS_bradyrigid_contra'
    output_file_path = join(FIG_PAPER, 'Figure_S3', "H___output.txt")
    with open(output_file_path, "w") as output_file:
        df_corr = get_correlation_df(df_norm, y,
                                     n_perm=N_PERM_CORR,
                                     use_peak_power=True, bands=BANDS,
                                     output_file=output_file)
    df_corr['kind'] = 'normalized'
    barplot_UPDRS_bands(df_corr, fig_dir='Figure_S3', prefix='H__',
                        fontsize_stat=8)
