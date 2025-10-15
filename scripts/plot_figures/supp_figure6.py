from os.path import join

import config as cfg

from scripts.plot_figures.settings import get_dfs
from scripts.utils_plot import _get_study_df, study_comparison


def supp_figure6(df_orig):
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df_abs = dataframes['df_abs']
    df_studies = _get_study_df(df_abs)
    study_comparison(df_studies, save_dir=join(cfg.FIG_PAPER, 'Figure_S6'))
