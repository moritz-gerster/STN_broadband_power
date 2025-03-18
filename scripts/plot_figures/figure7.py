import seaborn as sns

from scripts.plot_figures._correlation_scatter_within import (
    absolute_gamma_within, aperiodic_within)
from scripts.plot_figures._exemplary_subjects import exemplary_broadband_shifts
from scripts.plot_figures._peak_probability import (barplot_peaks,
                                                    get_peak_probability)
from scripts.plot_figures.settings import get_dfs, sns_darkgrid


def figure7(df_orig):
    # equalize subject count for model comparisons
    dataframes_equal = get_dfs(df_orig, ch_choice='ch_dist_sweet',
                               equalize_subjects_norm_abs=True)
    df_abs = dataframes_equal['df_abs']
    df_per = dataframes_equal['df_per']
    figure7a(df_abs)
    figure7b(df_per)
    figure7c(df_per)
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        figure7d(df_per)


def figure7a(df_abs):
    """Best absolute biomarker."""
    absolute_gamma_within(df_abs, fig_dir='Figure7', prefix='A__')


def figure7b(df_per):
    """Best absolute biomarker."""
    df_peaks = get_peak_probability(df_per)
    barplot_peaks(df_peaks, ylim=(0, 100), fig_dir='Figure7', prefix='B__')


def figure7c(df_per):
    """Best aperiodic biomarker."""
    aperiodic_within(df_per, fig_dir='Figure7', prefix='C__')


def figure7d(df_per):
    """Exemplary broadbandshifts."""
    exemplary_broadband_shifts(df_per, fig_dir='Figure7', prefix='D__')