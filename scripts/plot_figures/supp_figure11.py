from scripts.plot_figures._correlation_scatter import \
    representative_scatter_plot
from scripts.plot_figures._simulations import simulate_gamma_vs_broadband
from scripts.plot_figures.settings import get_dfs


def supp_figure11(df_orig):
    # equalize subject count for model comparisons
    dataframes_equal = get_dfs(df_orig, ch_choice='ch_dist_sweet',
                               equalize_subjects_norm_abs=True)
    df_abs = dataframes_equal['df_abs']
    figure11abcdef()
    figure11g(df_abs)
    figure11h(df_abs)
    figure11i(df_abs)


def figure11abcdef():
    """Best absolute biomarker."""
    simulate_gamma_vs_broadband(fig_dir='Figure_S11', prefix='A-F__')


def figure11g(df_abs):
    """Correlation mid gamma vs aperiodic parameters."""
    x = 'fm_offset_log'
    y = 'gamma_mid_abs_max5Hz_log'
    cond = 'off'
    representative_scatter_plot(df_abs, x, y, cond, fig_dir='Figure_S11',
                                average_hemispheres=True, xlabel=x,
                                figsize=(2.3, 1.4), prefix='G__')


def figure11h(df_abs):
    """Correlation mid gamma vs aperiodic parameters."""
    x = 'fm_exponent'
    y = 'gamma_mid_abs_max5Hz_log'
    cond = 'off'
    representative_scatter_plot(df_abs, x, y, cond, fig_dir='Figure_S11',
                                average_hemispheres=True, xlabel=x,
                                figsize=(2.3, 1.4), prefix='H__')


def figure11i(df_abs):
    """Correlation mid gamma vs aperiodic parameters."""
    x = 'full_fm_band_aperiodic_log'
    y = 'gamma_mid_abs_max5Hz_log'
    cond = 'off'
    representative_scatter_plot(df_abs, x, y, cond, fig_dir='Figure_S11',
                                average_hemispheres=True, xlabel=x,
                                figsize=(2.3, 1.4), prefix='I__')
