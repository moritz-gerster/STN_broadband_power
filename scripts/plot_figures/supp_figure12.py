from os import makedirs
from os.path import join

import seaborn as sns

from scripts.config import FIG_PAPER
from scripts.plot_figures._correlation_by_frequencies import (
    df_corr_freq, plot_psd_updrs_correlation)
from scripts.plot_figures._correlation_scatter import \
    representative_scatter_plot
from scripts.plot_figures._correlation_scatter_within import \
    aperiodic_within_by_age
from scripts.plot_figures.settings import (XTICKS_FREQ_high, get_dfs,
                                           sns_darkgrid)


def supp_figure12(df_orig):
    # equalize subject count for model comparisons
    dataframes_equal = get_dfs(df_orig, ch_choice='ch_dist_sweet',
                               equalize_subjects_norm_abs=True)
    df_abs = dataframes_equal['df_abs']
    df_per = dataframes_equal['df_per']
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        supp_figure11a(df_abs)
    supp_figure11bcd(df_abs)
    supp_figure11ef(df_per)


def supp_figure11a(df_abs):
    """Correlation age vs. power by frequency bin."""
    x = 'psd_log'
    y = 'patient_age'
    corr_method = 'spearman'
    data = df_abs[(df_abs.cond == 'off') & (df_abs.project == 'all')]
    df_freqs = df_corr_freq(data, x, y, corr_method=corr_method,
                            average_hemispheres=True, xmax=200)

    output_file_path = join(FIG_PAPER, 'Figure_S12', "A___output.txt")
    makedirs(join(FIG_PAPER, "Figure_S12"), exist_ok=True)
    xticks = XTICKS_FREQ_high + [150, 200]
    xticklabels = ['', 4, '', 13, '', 30, 45, 60, 100] + [150, 200]
    with open(output_file_path, "w") as output_file:
        plot_psd_updrs_correlation(df_freqs, x, y, 'absolute',
                                   fig_dir='Figure_S12', prefix='A__',
                                   xticks=xticks,
                                   xticklabels=xticklabels,
                                   figsize=(2.5, 1.3),
                                   xlabel=False, output_file=output_file)


def supp_figure11bcd(df_abs):
    """Correlation age vs aperiodic parameters."""
    x = 'patient_age'
    y = 'fm_offset_log'
    cond = 'off'
    representative_scatter_plot(df_abs, x, y, cond, fig_dir='Figure_S12',
                                average_hemispheres=True, xlabel=x,
                                figsize=(1.5, 1.4), prefix='B__')

    y = 'fm_exponent'
    representative_scatter_plot(df_abs, x, y, cond, fig_dir='Figure_S12',
                                average_hemispheres=True, xlabel=x,
                                figsize=(1.5, 1.4), prefix='C__')

    y = 'full_fm_band_aperiodic_log'
    representative_scatter_plot(df_abs, x, y, cond, fig_dir='Figure_S12',
                                average_hemispheres=True, xlabel=x,
                                figsize=(1.5, 1.4), prefix='D__')


def supp_figure11ef(df_per):
    """Within-patient correlation by age."""
    aperiodic_within_by_age(df_per, fig_dir='Figure_S12', prefix='EF__')