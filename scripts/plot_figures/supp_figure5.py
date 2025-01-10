from scripts.plot_figures._stun_effect import (power_vs_recovery,
                                               pre_post_vs_recovery,
                                               pre_post_vs_symptoms,
                                               updrs_pre_post)
from scripts.plot_figures.settings import get_dfs


def supp_figure5(df_orig):
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df_norm = dataframes['df_norm']
    supp_figure5a(df_norm)
    supp_figure5b(df_norm)
    supp_figure5c(df_norm)
    supp_figure5d(df_norm)


def supp_figure5a(df_norm):
    updrs_pre_post(df_norm, fig_dir='Figure_S5', prefix='A__')


def supp_figure5b(df_norm):
    pre_post_vs_recovery(df_norm, fig_dir='Figure_S5', prefix='B__')


def supp_figure5c(df_norm):
    power_vs_recovery(df_norm, fig_dir='Figure_S5', prefix='C__',
                      output_file=None)


def supp_figure5d(df_norm):
    pre_post_vs_symptoms(df_norm, fig_dir='Figure_S5', prefix='D__')