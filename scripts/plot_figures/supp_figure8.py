from scripts import config as cfg
from scripts.plot_figures._correlation_scatter_within import \
    periodic_gamma_within
from scripts.plot_figures._hemisphere_comparison import (
    gamma_peaks_by_hemisphere, normalized_bands_by_hemisphere,
    periodic_bands_by_hemisphere)
from scripts.plot_figures.settings import get_dfs


def supp_figure8(df_orig):
    # reproduction for adjacent channels (as Shreve 2017)
    dataframes_adjacent = get_dfs(df_orig,
                                  ch_choice='ch_adj_beta_high_max_off',
                                  equalize_subjects_norm_abs=True)
    df_per_adj = dataframes_adjacent['df_per']
    df_norm_adj = dataframes_adjacent['df_norm']

    # original channel choice (distant channels)
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df_per_orig = dataframes['df_per']
    df_norm_orig = dataframes['df_norm']

    # # more exact reproduction of Shreve 2017 (choosing channels based on
    # # maximum alpha-beta power, same result though):
    # dataframes_abmax = get_dfs(df_orig,
    #                            ch_choice='ch_chmax_alpha_beta_abs_max_log')
    # df_norm = dataframes_abmax['df_norm']
    # df_per_abmax = dataframes_abmax['df_per']
    # chs_adj = ['LFP_1-2', 'LFP_2-3', 'LFP_3-4']
    # df_norm_adj = df_norm[df_norm.ch.isin(chs_adj)]
    # df_per_adj = df_per_abmax[df_per_abmax.ch.isin(chs_adj)]

    supp_figure8b(df_norm_adj)
    supp_figure8c(df_per_adj)
    supp_figure8d(df_per_adj)  # original channel choice
    supp_figure8f(df_norm_orig)
    supp_figure8g(df_per_orig)
    supp_figure8h(df_per_orig)  # original channel choice


def supp_figure8b(df_norm_adj, prefix='B__'):
    """Reproduce Shreve et al."""
    normalized_bands_by_hemisphere(df_norm_adj, fig_dir='Figure_S8',
                                   prefix=prefix)


def supp_figure8c(df_per_adj, prefix='C__'):
    """S7e for periodic power."""
    periodic_bands_by_hemisphere(df_per_adj, fig_dir='Figure_S8',
                                 prefix=prefix)


def supp_figure8d(df_per):
    """Periodic gamma does correlate within subjects for adjacent LFP
    channels."""
    exemplary_subs = cfg.EXEMPLARY_SUBS_GAMMA
    periodic_gamma_within(df_per, fig_dir='Figure_S8', prefix='D1__',
                          exemplary_subs=exemplary_subs, figsize=(1.55, 1.34),
                          bbox_to_anchor=(-.3, 1))
    gamma_peaks_by_hemisphere(df_per, fig_dir='Figure_S8', prefix='D2__')


def supp_figure8f(df_norm_dist, prefix='F__'):
    """Reproduce Shreve et al."""
    normalized_bands_by_hemisphere(df_norm_dist, fig_dir='Figure_S8',
                                   prefix=prefix)


def supp_figure8g(df_per_dist, prefix='G__'):
    """S7e for periodic power."""
    periodic_bands_by_hemisphere(df_per_dist, fig_dir='Figure_S8',
                                 prefix=prefix)


def supp_figure8h(df_per):
    """Periodic gamma does not correlate within subjects for distant LFP
    channels."""
    exemplary_subs = cfg.EXEMPLARY_SUBS_GAMMA
    periodic_gamma_within(df_per, fig_dir='Figure_S8', prefix='H1__',
                          exemplary_subs=exemplary_subs, figsize=(1.55, 1.34),
                          bbox_to_anchor=(-.3, 1))
    gamma_peaks_by_hemisphere(df_per, fig_dir='Figure_S8', prefix='H2__')
