from os import makedirs
from os.path import join

import seaborn as sns

from scripts.config import FIG_PAPER
from scripts.plot_figures._correlation_by_frequencies import corr_by_freq_bars
from scripts.plot_figures._correlation_scatter import plot_all
from scripts.plot_figures._exemplary_spectra import representative_spectrum
from scripts.plot_figures._power_spectra import plot_abs_per_spectra
from scripts.plot_figures._simulations import simulate_all
from scripts.plot_figures.settings import (N_BOOT_COHEN, N_PERM_CLUSTER,
                                           get_dfs, sns_darkgrid)


def figure3(df_orig):
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df_abs = dataframes['df_abs']
    df_abs = df_abs[(df_abs.project != 'all')]

    df_per = dataframes['df_per']
    df_per = df_per[(df_per.project != 'all')]
    with sns.axes_style('darkgrid', rc=sns_darkgrid):
        figure3a(df_abs)
        figure3b_f()
        figure3c(dataframes)
        figure3d1(dataframes)
        figure3e(df_per)
        figure3g(dataframes)
        figure3h1(dataframes)
    figure3d2(df_abs)
    figure3h2(df_per)


def figure3a(df_abs):
    """Exemplary spectrum absolute."""
    output_file_path = join(FIG_PAPER, 'Figure3', "A___output.txt")
    makedirs(join(FIG_PAPER, 'Figure3'), exist_ok=True)
    with open(output_file_path, "w") as output_file:
        representative_spectrum(df_abs, 'absolute',
                                fig_dir='Figure3',
                                legend=True, yscale='linear',
                                ylabel=False, xlabel=False,
                                figsize=(1.77, 1.3),
                                leg_kwargs={'ncol': 2},
                                output_file=output_file,
                                prefix='A1__')
        representative_spectrum(df_abs, 'absolute',
                                fig_dir='Figure3',
                                legend=False,
                                xlabel=False,
                                figsize=(1.65, 1.33),
                                output_file=output_file,
                                xscale='log', ylabel=False,
                                prefix='A2__')


def figure3b_f():
    """Simulations for subfigures b and g."""
    output_file_path = join(FIG_PAPER, 'Figure3', "B+F___output.txt")
    with open(output_file_path, "w") as output_file:
        simulate_all(fig_dir='Figure3', output_file=output_file)


def figure3c(dataframes):
    """Absolute power spectra by Levodopa and severity."""
    plot_abs_per_spectra(dataframes, 'absolute', fig_dir='Figure3',
                         prefix='C__', height_star=.985,
                         n_perm=N_PERM_CLUSTER, n_boot=N_BOOT_COHEN
                         )


def figure3g(dataframes):
    """Periodic power spectra by Levodopa and severity."""
    plot_abs_per_spectra(dataframes, 'periodic', fig_dir='Figure3',
                         prefix='G__', height_star=.985,
                         n_perm=N_PERM_CLUSTER, n_boot=N_BOOT_COHEN
                         )


def figure3d1(dataframes):
    """Correlation by frequency bin."""
    output_file_path = join(FIG_PAPER, 'Figure3', "D1___output.txt")
    with open(output_file_path, "w") as output_file:
        corr_by_freq_bars(dataframes, ['absolute'], output_file=output_file,
                          y='UPDRS_III', ylabel=False, conds=['off'],
                          fig_dir='Figure3', prefix='D1__')


def figure3d2(df_abs):
    """Correlation by canonical frequency."""
    X = ['theta_abs_mean_log',
         'beta_low_abs_mean_log',
         'gamma_low_abs_mean_log']
    df_abs = df_abs[(df_abs.cond == 'off')]
    output_file_path = join(FIG_PAPER, 'Figure3', "D2___output.txt")
    with open(output_file_path, "w") as output_file:
        plot_all(df_abs, X, 'UPDRS_III', 'absolute', fig_dir='Figure3',
                 prefix='D2__', output_file=output_file, ylabel=False)


def figure3h1(dataframes):
    """Correlation by frequency bin."""
    output_file_path = join(FIG_PAPER, 'Figure3', "H1___output.txt")
    with open(output_file_path, "w") as output_file:
        corr_by_freq_bars(dataframes, ['periodic'], output_file=output_file,
                          y='UPDRS_III', conds=['off'], ylabel=False,
                          ylim=(-.34, .31), fig_dir='Figure3', prefix='H1__')


def figure3h2(df_per):
    """Correlation by canonical frequency."""
    X = ['fm_offset_log', 'beta_low_fm_mean_log', 'gamma_low_fm_mean_log']
    df_per = df_per[(df_per.cond == 'off')]
    output_file_path = join(FIG_PAPER, 'Figure3', "H2___output.txt")
    with open(output_file_path, "w") as output_file:
        plot_all(df_per, X, 'UPDRS_III', 'periodic', fig_dir='Figure3',
                 prefix='H2__', output_file=output_file, ylabel=False)


def figure3e(df_per):
    """Exemplary spectrum periodic."""
    representative_spectrum(df_per, 'periodic',
                            fig_dir='Figure3',
                            legend=False, yscale='linear',
                            figsize=(1.77, 1.3), ylabel=False, xlabel=False,
                            prefix='E1__')
    representative_spectrum(df_per, 'periodic',
                            fig_dir='Figure3',
                            legend=False,
                            figsize=(1.56, 1.33),
                            xscale='log', ylabel=False, xlabel=False,
                            prefix='E2__')
