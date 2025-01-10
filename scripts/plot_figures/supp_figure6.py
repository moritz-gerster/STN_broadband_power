from scripts.plot_figures._aperiodic_off_vs_on import band_barplot
from scripts.plot_figures._correlation_scatter import corr_offset_theta
from scripts.plot_figures.settings import get_dfs


def supp_figure6(df_orig):
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df_abs = dataframes['df_abs']
    df_per = dataframes['df_per']
    supp_figure6a(df_abs)
    supp_figure6b(df_per)
    supp_figure6c(df_per)


def supp_figure6a(df_abs):
    kind = 'absolute'
    ycols = ['psd_sum_5to95']
    band_barplot(df_abs, kind, ycols, fig_dir='Figure_S6', prefix='A__',
                 projects=['all'], figsize=(1.5, 1.5),
                 xticklabels=['Spectral sum 5-95 Hz'])


def supp_figure6b(df_per):
    kind = 'periodicAP'
    ycols = ['fm_exponent', 'fm_offset_log', 'full_fm_band_aperiodic_log']
    band_barplot(df_per, kind, ycols, fig_dir='Figure_S6', prefix='B__',
                 projects=['all'], figsize=(2, 1.5),
                 xticklabels=['1/f exponent', 'Offset', 'Aper. power'])


def supp_figure6c(df_per):
    corr_offset_theta(df_per, fig_dir='Figure_S6', prefix='C__')