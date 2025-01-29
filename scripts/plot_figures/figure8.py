import seaborn as sns

from scripts.plot_figures._correlation_scatter_within import (
    absolute_gamma_within, aperiodic_within)
from scripts.plot_figures._exemplary_subjects import exemplary_broadband_shifts
from scripts.plot_figures._peak_probability import (barplot_peaks,
                                                    get_peak_probability)
from scripts.plot_figures.settings import get_dfs


def figure8(df_orig):
    # equalize subject count for model comparisons
    dataframes_equal = get_dfs(df_orig, ch_choice='ch_dist_sweet',
                               equalize_subjects_norm_abs=True)
    df_abs = dataframes_equal['df_abs']
    df_per = dataframes_equal['df_per']
    figure8a(df_abs)
    figure8b(df_per)
    figure8c(df_per)
    with sns.axes_style('darkgrid'):
        figure8d(df_per)


def figure8a(df_abs):
    """Best absolute biomarker."""
    absolute_gamma_within(df_abs, fig_dir='Figure8', prefix='A__')


def figure8b(df_per):
    """Best absolute biomarker."""
    df_peaks = get_peak_probability(df_per)
    barplot_peaks(df_peaks, ylim=(0, 100), fig_dir='Figure8', prefix='B__')


def figure8c(df_per):
    """Best aperiodic biomarker."""
    aperiodic_within(df_per, fig_dir='Figure8', prefix='C__')


def figure8d(df_per):
    """Exemplary broadbandshifts."""
    exemplary_broadband_shifts(df_per, fig_dir='Figure8', prefix='D__')
