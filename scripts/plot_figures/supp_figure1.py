from os.path import join

import seaborn as sns

from scripts.config import FIG_PAPER
from scripts.plot_figures._power_spectra import plot_absolute_spectra
from scripts.plot_figures.settings import get_dfs
from scripts.utils_plot import (_dataset_dbs_models_leads,
                                _mni_coords_datasets, _patient_symptoms_flat)


def supp_figure1(df_orig):
    """Plot and save supplementary figure 1."""
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df = dataframes['df']
    df_abs = dataframes['df_abs']
    supp_figure1a(df)
    supp_figure1b()
    with sns.axes_style('darkgrid'):
        supp_figure1c(df_abs)
    supp_figure1d(df)


def supp_figure1a(df):
    """DBS lead models by dataset."""
    save_dir = join(FIG_PAPER, "Figure_S1")
    _dataset_dbs_models_leads(df, save=True, save_dir=save_dir, prefix='A__')


def supp_figure1b():
    """DBS lead positions by dataset."""
    _mni_coords_datasets(fig_dir='Figure_S1', prefix='B__')


def supp_figure1c(df_abs):
    """Absolute power spectra by dataset."""
    plot_absolute_spectra(df_abs, fig_dir='Figure_S1', prefix='C__')


def supp_figure1d(df):
    """UPDRS symptoms by Levodopa condition and dataset."""
    _patient_symptoms_flat(df, fig_dir='Figure_S1', conds=['off'],
                           prefix='D1__')
    _patient_symptoms_flat(df, fig_dir='Figure_S1', conds=['on'],
                           prefix='D2__')
    _patient_symptoms_flat(df, fig_dir='Figure_S1', conds=['offon_abs'],
                           prefix='D3__')
