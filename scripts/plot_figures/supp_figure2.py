from os import makedirs
from os.path import join

import seaborn as sns

from scripts.config import FIG_PAPER
from scripts.plot_figures._correlation_by_bands import barplot_UPDRS_bands
from scripts.plot_figures._correlation_by_frequencies import (
    df_corr_freq, plot_psd_updrs_correlation)
from scripts.plot_figures._correlation_scatter import \
    representative_scatter_plot
from scripts.plot_figures._forrest_plot_datasets_correlation import \
    forest_plot_correlation
from scripts.plot_figures._psd_clusters import plot_psd_by_severity_conds
from scripts.plot_figures._table2_freq_bands_literature import plot_beta_ranges
from scripts.plot_figures.settings import (BANDS, N_PERM_CORR, XTICKS_FREQ_low,
                                           XTICKS_FREQ_low_labels, get_dfs,
                                           sns_darkgrid)
from scripts.utils import get_correlation_df


def supp_figure2(df_orig):
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df_norm = dataframes['df_norm']
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        supp_figure2a()
    supp_figure2b(df_norm)
    supp_figure2c(df_norm)
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        supp_figure2d(dataframes)
        supp_figure2e(df_norm)
    supp_figure2f(df_norm)
    supp_figure2g(df_norm)
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        supp_figure2h(dataframes)
        supp_figure2i(df_norm)
    supp_figure2j(df_norm)
    supp_figure2k(df_norm)


def supp_figure2a():
    """Beta frequency ranges from literature."""
    plot_beta_ranges(fig_dir='Figure_S2', prefix='A__', xlabel=False)


def supp_figure2b(df_norm):
    """Beta ~ UPDRS-III correlation by dataset."""
    cond = 'off'
    df_cond = df_norm[df_norm.cond == cond]
    bands = ['alpha_beta_abs_mean_log',
             'beta_abs_mean_log',
             'beta_low_abs_mean_log']
    output_file_path = join(FIG_PAPER, 'Figure_S2', "B___output.txt")

    with open(output_file_path, "w") as output_file:
        forest_plot_correlation(df_cond, bands, 'UPDRS_III',
                                fig_dir='Figure_S2', prefix='B__',
                                dataset_labels=True, title_y=.95,
                                figsize=(3, 1.15),
                                xlabel=False, output_file=output_file)


def supp_figure2c(df_norm):
    """Beta ~ Bradykinesia-Rigdity correlation by dataset."""
    cond = 'off'
    df_cond = df_norm[df_norm.cond == cond]
    bands = ['alpha_beta_abs_mean_log',
             'beta_abs_mean_log',
             'beta_low_abs_mean_log']
    output_file_path = join(FIG_PAPER, 'Figure_S2', "C___output.txt")

    with open(output_file_path, "w") as output_file:
        forest_plot_correlation(df_cond, bands, 'UPDRS_bradyrigid_contra',
                                fig_dir='Figure_S2', prefix='C__',
                                dataset_labels=False, title_y=.95,
                                figsize=(3, 1.15),
                                output_file=output_file, xlabel=False)


def supp_figure2d(dataframes):
    """PSDs by severity conditions."""
    output_file_path = join(FIG_PAPER, 'Figure_S2', "D___output.txt")
    makedirs(join(FIG_PAPER, 'Figure_S2'), exist_ok=True)
    with open(output_file_path, "w") as output_file:
        plot_psd_by_severity_conds(dataframes, 'normalized', ['off'],
                                   lateralized_updrs=False,
                                   color_by_kind=False,
                                   xmin=2, xmax=45, info_title=False,
                                   figsize=(1.9, 1.3295), fig_dir='Figure_S2',
                                   xticks=XTICKS_FREQ_low,
                                   xticklabels=XTICKS_FREQ_low_labels,
                                   ylim=(0, 8), prefix='D__',
                                   output_file=output_file)


def supp_figure2e(df_norm):
    """Correlation by frequency bin."""
    x = 'psd_log'
    y = 'UPDRS_III'
    corr_method = 'spearman'
    data = df_norm[(df_norm.cond == 'off')]
    df_freqs = df_corr_freq(data, x, y, corr_method=corr_method)

    output_file_path = join(FIG_PAPER, 'Figure_S2', "E___output.txt")
    with open(output_file_path, "w") as output_file:
        plot_psd_updrs_correlation(df_freqs, x, y, 'normalized',
                                   fig_dir='Figure_S2', prefix='E__',
                                   output_file=output_file)


def supp_figure2f(df_norm):
    """Plot correlations by bands."""
    total_power = True
    cond = 'off'
    df_plot = df_norm[(df_norm.cond == cond)]
    y = 'UPDRS_III'
    output_file_path = join(FIG_PAPER, 'Figure_S2', "F___output.txt")
    with open(output_file_path, "w") as output_file:
        df_corr = get_correlation_df(df_plot, y, total_power=total_power,
                                     n_perm=N_PERM_CORR,
                                     use_peak_power=True, bands=BANDS,
                                     output_file=output_file)
    df_corr['kind'] = 'normalized'
    barplot_UPDRS_bands(df_corr, fig_dir='Figure_S2', prefix='F__',
                        fontsize_stat=8)


def supp_figure2g(df_norm):
    """Representative correlation scatter plot."""
    x = 'beta_low_abs_mean_log'
    y = 'UPDRS_III'
    cond = 'off'
    representative_scatter_plot(df_norm, x, y, cond, fig_dir='Figure_S2',
                                figsize=(1.3, 1.4), prefix='G__')


def supp_figure2h(dataframes):
    """PSDs by severity conditions."""
    output_file_path = join(FIG_PAPER, 'Figure_S2', "H___output.txt")
    makedirs(join(FIG_PAPER, 'Figure_S2'), exist_ok=True)
    with open(output_file_path, "w") as output_file:
        plot_psd_by_severity_conds(dataframes, 'normalized', ['off'],
                                   lateralized_updrs=True, color_by_kind=False,
                                   xmin=2, xmax=45, info_title=False,
                                   figsize=(1.9, 1.3295), fig_dir='Figure_S2',
                                   xticks=XTICKS_FREQ_low,
                                   xticklabels=XTICKS_FREQ_low_labels,
                                   ylim=(0, 8), prefix='H__',
                                   output_file=output_file)


def supp_figure2i(df_norm):
    """Correlation by frequency bin."""
    x = 'psd_log'
    y = 'UPDRS_bradyrigid_contra'
    corr_method = 'spearman'
    data = df_norm[(df_norm.cond == 'off')]
    df_freqs = df_corr_freq(data, x, y, corr_method=corr_method)

    output_file_path = join(FIG_PAPER, 'Figure_S2', "I___output.txt")
    with open(output_file_path, "w") as output_file:
        plot_psd_updrs_correlation(df_freqs, x, y, 'normalized',
                                   fig_dir='Figure_S2', prefix='I__',
                                   output_file=output_file)


def supp_figure2j(df_norm):
    """Plot correlations by bands."""
    total_power = True
    cond = 'off'
    df_plot = df_norm[(df_norm.cond == cond)]
    y = 'UPDRS_bradyrigid_contra'
    output_file_path = join(FIG_PAPER, 'Figure_S2', "J___output.txt")
    with open(output_file_path, "w") as output_file:
        df_corr = get_correlation_df(df_plot, y, total_power=total_power,
                                     n_perm=N_PERM_CORR,
                                     use_peak_power=True, bands=BANDS,
                                     output_file=output_file)
    df_corr['kind'] = 'normalized'
    barplot_UPDRS_bands(df_corr, fig_dir='Figure_S2', prefix='J__',
                        fontsize_stat=8)


def supp_figure2k(df_norm):
    """Representative correlation scatter plot."""
    x = 'beta_low_abs_mean_log'
    y = 'UPDRS_bradyrigid_contra'
    cond = 'off'
    representative_scatter_plot(df_norm, x, y, cond, fig_dir='Figure_S2',
                                figsize=(1.3, 1.4), prefix='K__')
