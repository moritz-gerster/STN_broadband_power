from scripts.plot_figures._exemplary_spectra import (
    exemplary_spectrum_mini, exemplary_spectrum_mini_kinds,
    exemplary_time_series)
from scripts.plot_figures._explain_repeated_measures import \
    repeated_measures_simple
from scripts.plot_figures.settings import get_dfs


def figure1(df_orig):
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df_abs = dataframes['df_abs']
    figure1b(df_abs)
    figure1c(dataframes)


def figure1b(df_abs):
    """Introductory subfigure created externally."""
    exemplary_time_series(fig_dir='Figure1', prefix='B1__')
    exemplary_spectrum_mini(df_abs, fig_dir='Figure1', prefix='B2__')


def figure1c(dataframes):
    """Plot toy example within vs across correlations."""
    exemplary_spectrum_mini_kinds(dataframes, fig_dir='Figure1', prefix='C2__')
    repeated_measures_simple(fig_dir='Figure1', prefix='C3__')
