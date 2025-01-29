from os.path import join

import seaborn as sns

from scripts.config import FIG_PAPER
from scripts.plot_figures._exemplary_spectra import representative_spectrum
from scripts.plot_figures._power_spectra import plot_normalized_spectra
from scripts.plot_figures.settings import get_dfs
from scripts.utils_plot import _dataset_comparison_divided


def figure1(df_orig):
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df = dataframes['df']
    df_norm = dataframes['df_norm']
    figure1a()
    with sns.axes_style('darkgrid'):
        figure1b(df_norm)
        figure1c(df_norm)
    figure1d(df)


def figure1a():
    """Introductory subfigure created externally."""
    pass


def figure1b(df_norm):
    """Exemplary spectrum normalized."""
    representative_spectrum(df_norm, 'normalized',
                            fig_dir='Figure1',
                            legend=True, yscale='linear',
                            height=1.7, aspect=.8,
                            prefix='B1__')
    representative_spectrum(df_norm, 'normalized',
                            fig_dir='Figure1',
                            legend=False,
                            height=1.7, aspect=.8,
                            xscale='log', ylabel='',
                            prefix='B2__')


def figure1c(df_norm):
    """Normalized power spectra by dataset."""
    plot_normalized_spectra(df_norm, prefix='C__')


def figure1d(df):
    """Dataset meta infos."""
    save_dir = join(FIG_PAPER, "Figure1")
    _dataset_comparison_divided(df, save=True, save_dir=save_dir, prefix='D__')
