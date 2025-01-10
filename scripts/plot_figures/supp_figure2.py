from os import makedirs
from os.path import join

import seaborn as sns

from scripts.config import FIG_PAPER
from scripts.plot_figures._correlation_by_bands import barplot_UPDRS_bands
from scripts.plot_figures._correlation_by_frequencies import (
    df_corr_freq, plot_psd_updrs_correlation)
from scripts.plot_figures._correlation_scatter import \
    representative_scatter_plot
from scripts.plot_figures._psd_clusters import plot_psd_by_severity_conds
from scripts.plot_figures.settings import (BANDS, N_PERM_CORR, XTICKS_FREQ_low,
                                           XTICKS_FREQ_low_labels, get_dfs)
from scripts.utils import get_correlation_df


def supp_figure2(df_orig):
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df_norm = dataframes['df_norm']
    with sns.axes_style('darkgrid'):
        supp_figure2a(dataframes)
        supp_figure2b(df_norm)
    supp_figure2c(df_norm)
    supp_figure2d(df_norm)


def supp_figure2a(dataframes):
    """PSDs by severity conditions."""
    output_file_path = join(FIG_PAPER, 'Figure_S2', "A___output.txt")
    makedirs(join(FIG_PAPER, 'Figure_S2'), exist_ok=True)
    with open(output_file_path, "w") as output_file:
        plot_psd_by_severity_conds(dataframes, 'normalized', ['off'],
                                   lateralized_updrs=True, color_by_kind=False,
                                   xmin=2, xmax=45, info_title=False,
                                   figsize=(2, 1.3), fig_dir='Figure_S2',
                                   xticks=XTICKS_FREQ_low,
                                   xticklabels=XTICKS_FREQ_low_labels,
                                   ylim=(0, 8), prefix='A__',
                                   output_file=output_file)


def supp_figure2b(df_norm):
    """Correlation by frequency bin."""
    x = 'psd_log'
    y = 'UPDRS_bradyrigid_contra'
    corr_method = 'spearman'
    data = df_norm[(df_norm.cond == 'off')]
    df_freqs = df_corr_freq(data, x, y, corr_method=corr_method)

    output_file_path = join(FIG_PAPER, 'Figure_S2', "B___output.txt")
    with open(output_file_path, "w") as output_file:
        plot_psd_updrs_correlation(df_freqs, x, y, 'normalized',
                                   fig_dir='Figure_S2', prefix='B__',
                                   output_file=output_file)


def supp_figure2c(df_norm):
    """Plot correlations by bands."""
    total_power = True
    cond = 'off'
    df_plot = df_norm[(df_norm.cond == cond)]
    y = 'UPDRS_bradyrigid_contra'
    output_file_path = join(FIG_PAPER, 'Figure_S2', "C___output.txt")
    with open(output_file_path, "w") as output_file:
        df_corr = get_correlation_df(df_plot, y, total_power=total_power,
                                     n_perm=N_PERM_CORR,
                                     use_peak_power=True, bands=BANDS,
                                     output_file=output_file)
    df_corr['kind'] = 'normalized'
    barplot_UPDRS_bands(df_corr, fig_dir='Figure_S2', prefix='C__',
                        figsize=(1.9, 1.3))


def supp_figure2d(df_norm):
    """Representative correlation scatter plot."""
    x = 'beta_low_abs_mean_log'
    y = 'UPDRS_bradyrigid_contra'
    cond = 'off'
    representative_scatter_plot(df_norm, x, y, cond, fig_dir='Figure_S2',
                                prefix='D__')