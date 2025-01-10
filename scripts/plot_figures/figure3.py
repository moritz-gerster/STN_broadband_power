from os.path import join

import seaborn as sns

from scripts.config import FIG_PAPER
from scripts.plot_figures._correlation_by_frequencies import figure2
from scripts.plot_figures._correlation_scatter import plot_all
from scripts.plot_figures._exemplary_spectra import representative_spectrum
from scripts.plot_figures._power_spectra import plot_abs_per_spectra
from scripts.plot_figures._simulations import simulate_all
from scripts.plot_figures.settings import get_dfs


def figure3(df_orig):
    dataframes = get_dfs(df_orig, ch_choice='ch_dist_sweet')
    df_abs = dataframes['df_abs']
    df_abs = df_abs[(df_abs.cond == 'off') & (df_abs.project != 'all')]

    df_per = dataframes['df_per']
    df_per = df_per[(df_per.cond == 'off') & (df_per.project != 'all')]
    with sns.axes_style('darkgrid'):
        figure3a(df_abs)
        figure3b_g()
        figure3c_d(dataframes)
        figure3h_i(dataframes)
        figure3e1(dataframes)
        figure3f(df_per)
        figure3j1(dataframes)
    figure3e2(df_abs)
    figure3j2(df_per)


def figure3a(df_abs):
    """Exemplary spectrum absolute."""
    representative_spectrum(df_abs, 'absolute',
                            fig_dir='Figure3',
                            legend=True, yscale='linear',
                            height=1.5, aspect=1,
                            prefix='A1__')
    representative_spectrum(df_abs, 'absolute',
                            fig_dir='Figure3',
                            legend=False,
                            height=1.5, aspect=1,
                            xscale='log', ylabel='',
                            prefix='A2__')


def figure3b_g():
    """Simulations for subfigures b and g."""
    output_file_path = join(FIG_PAPER, 'Figure3', "BG___output.txt")
    with open(output_file_path, "w") as output_file:
        simulate_all(fig_dir='Figure3', output_file=output_file)


def figure3c_d(dataframes):
    """Absolute power spectra by Levodopa and severity."""
    plot_abs_per_spectra(dataframes, 'absolute', prefix='CD__')


def figure3h_i(dataframes):
    """Periodic power spectra by Levodopa and severity."""
    plot_abs_per_spectra(dataframes, 'periodic', prefix='HI__')


def figure3e1(dataframes):
    """Correlation by frequency bin."""
    output_file_path = join(FIG_PAPER, 'Figure3', "E1___output.txt")
    with open(output_file_path, "w") as output_file:
        figure2(dataframes, ['absolute'], output_file=output_file,
                y='UPDRS_III',
                conds=['off'],
                fig_dir='Figure3',
                prefix='E1__')


def figure3e2(df_abs):
    """Correlation by frequency bin."""
    X = ['theta_abs_mean_log',
         'beta_low_abs_mean_log',
         'gamma_low_abs_mean_log']
    output_file_path = join(FIG_PAPER, 'Figure3', "E2___output.txt")
    with open(output_file_path, "w") as output_file:
        plot_all(df_abs, X, 'UPDRS_III', 'absolute', fig_dir='Figure3',
                 prefix='E2__', output_file=output_file)


def figure3j1(dataframes):
    """Correlation by frequency bin."""
    output_file_path = join(FIG_PAPER, 'Figure3', "J1___output.txt")
    with open(output_file_path, "w") as output_file:
        figure2(dataframes, ['periodic'], output_file=output_file,
                y='UPDRS_III',
                conds=['off'],
                fig_dir='Figure3',
                prefix='J1__')


def figure3j2(df_per):
    """Correlation by frequency bin."""
    X = ['fm_offset_log', 'beta_low_fm_mean_log', 'gamma_low_fm_mean_log']
    output_file_path = join(FIG_PAPER, 'Figure3', "J2___output.txt")
    with open(output_file_path, "w") as output_file:
        plot_all(df_per, X, 'UPDRS_III', 'periodic', fig_dir='Figure3',
                 prefix='J2__', output_file=output_file)


def figure3f(df_per):
    """Exemplary spectrum periodic."""
    representative_spectrum(df_per, 'periodic',
                            fig_dir='Figure3',
                            legend=False, yscale='linear',
                            height=1.5, aspect=1,
                            prefix='F1__')
    representative_spectrum(df_per, 'periodic',
                            fig_dir='Figure3',
                            legend=True,
                            height=1.5, aspect=1,
                            xscale='log', ylabel='',
                            prefix='F2__')