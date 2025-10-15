from scripts.plot_figures._correlation_scatter_within import (
    absolute_gamma_within_tremor, aperiodic_within_tremor,
    absolute_gamma_within_hemi, aperiodic_within_hemi)
from scripts.plot_figures.settings import get_dfs


def supp_figure10(df_orig):
    # equalize subject count for model comparisons
    dataframes_equal = get_dfs(df_orig, ch_choice='ch_dist_sweet',
                               equalize_subjects_norm_abs=True)
    df_abs = dataframes_equal['df_abs']
    df_per = dataframes_equal['df_per']
    figure10a(df_abs)
    figure10b(df_per)
    figure10c(df_abs)
    figure10d(df_per)


def figure10a(df_abs):
    """Best absolute biomarker."""
    absolute_gamma_within_hemi(df_abs, fig_dir='Figure_S10', prefix='B__')


def figure10b(df_per):
    """Best aperiodic biomarker."""
    aperiodic_within_hemi(df_per, fig_dir='Figure_S10', prefix='C__')


def figure10c(df_abs):
    """Best absolute biomarker."""
    absolute_gamma_within_tremor(df_abs, fig_dir='Figure_S10', prefix='E__')


def figure10d(df_per):
    """Best aperiodic biomarker."""
    aperiodic_within_tremor(df_per, fig_dir='Figure_S10', prefix='F__')
