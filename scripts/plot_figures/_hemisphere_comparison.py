from os.path import join

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

import scripts.config as cfg
from scripts.plot_figures.settings import *
from scripts.utils_plot import _axes2d, _save_fig, _stat_anno


def gamma_peaks_by_hemisphere(df_per, fig_dir='Figure_S7', prefix=''):
    y = 'fm_peak_count'
    cond = 'on'
    col = f"patient_symptom_dominant_side_BR_{cond}"

    df_mild = df_per[(df_per[col] == 'mild side') & (df_per.cond == cond)]
    df_severe = df_per[(df_per[col] == 'severe side') & (df_per.cond == cond)]

    bands = ['gamma']
    n_col = len(bands)
    fig, axes = plt.subplots(1, n_col, figsize=(.8*n_col, 1.3), sharey=False)
    axes = _axes2d(axes, 1, n_col)
    for i, band in enumerate(bands):
        mild_percentage = ((df_mild[f'{band}_{y}'] > 0).sum()
                           / len(df_mild) * 100)
        severe_percentage = ((df_severe[f'{band}_{y}'] > 0).sum()
                             / len(df_severe) * 100)
        ax = axes[0, i]
        ax.bar(1, severe_percentage, color=cfg.COLOR_DIC['periodic2'])
        ax.bar(2, mild_percentage, color=cfg.COLOR_DIC['periodic2'])
        ax.set_ylabel(None)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.set_title(f'Has {cfg.BAND_NAMES_GREEK_SHORT[band]} peak')
        ax.set_xticks([1, 2], ['Severe', 'Mild'])
        ax.set_xlabel('Hemisphere')
    plt.tight_layout()
    _save_fig(fig, f'{prefix}{'_'.join(bands)}_{y}_{cond}',
              bbox_inches=None, transparent=True,
              save_dir=join(cfg.FIG_PAPER, fig_dir))


def beta_peaks_by_hemisphere(df_per, fig_dir='Figure_S7', prefix=''):
    conds  = ['off', 'on']
    bands = ['alpha_beta', 'beta']
    n_rows = 1  # len(conds)
    n_col = len(bands)

    y = 'fm_peak_count'

    fig, axes = plt.subplots(n_rows, n_col, figsize=(2, 1.5))
    axes = _axes2d(axes, n_rows, n_col)
    row = 0
    for cond in conds:
        col = f"patient_symptom_dominant_side_BR_{cond}"
        df_off = df_per[(df_per.cond == 'off')]
        df_on = df_per[(df_per.cond == 'on')]

        for col, band in enumerate(bands):
            off_percentage = (df_off[f'{band}_{y}'] > 0).sum() / len(df_off)
            on_percentage = (df_on[f'{band}_{y}'] > 0).sum() / len(df_on)
            ax = axes[row, col]
            bar1 = ax.bar(1, off_percentage, color=cfg.COLOR_DIC['periodic'])
            bar2 = ax.bar(2, on_percentage, color=cfg.COLOR_DIC['periodic2'])
            ax.bar_label(bar1, fmt='{:.1%}')
            ax.bar_label(bar2, fmt='{:.1%}')
            print(f'Peak in either hemisphere {cond}: {off_percentage:.2f}%')
            if row == 0:
                ax.set_title(f'Has {cfg.BAND_NAMES_GREEK_SHORT[band]} peak')
            ax.set_xticks([1, 2], ['off', 'on'])
    plt.tight_layout()
    _save_fig(fig, f'{prefix}reproduce_shreve_fittedPeaks',
              bbox_inches=None, transparent=True,
              save_dir=join(cfg.FIG_PAPER, fig_dir))


def normalized_bands_by_hemisphere(df_norm, fig_dir='Figure_S7', prefix=''):
    df_norm = df_norm[~df_norm.project.isin(['all'])]
    df_norm = df_norm[df_norm.cond.isin(['off'])
                      | ((df_norm.cond.isin(['on'])
                          & df_norm.dominant_side_consistent))]

    conds = ['off', 'on']
    yvals = ['alpha_beta_abs_mean', 'alpha_abs_mean',
             'beta_low_abs_mean', 'beta_high_abs_mean',
             'gamma_abs_mean']
    n_rows = len(conds)
    n_col = len(yvals)
    severity = ['severe side', 'mild side']

    fig, axes = plt.subplots(n_rows, n_col, figsize=(4.4, 1.5), sharey=False,
                             sharex=True)
    axes = _axes2d(axes, n_rows, n_col)
    for row, cond in enumerate(conds):

        x = f"patient_symptom_dominant_side_BR_{cond}"
        df_plot = df_norm[(df_norm.cond == cond)]
        # df_plot = df[(df.cond == cond)]

        for col, y in enumerate(yvals):
            ax = axes[row, col]
            # sns.stripplot(df_cond, y=y, x=hemi, ax=ax, size=1)
            kind = 'normalized' if row == 0 else 'normalized2'
            sns.pointplot(df_plot, y=y, x=x, ax=ax, color=cfg.COLOR_DIC[kind],
                          order=severity,
                          errorbar=('ci', 95))
            _stat_anno(ax, df_plot, x, y, fontsize=FONTSIZE_ASTERISK)

            if row == 0:
                ax.set_title(cfg.PLOT_LABELS_SHORT[y].replace(' mean', ''))
            ax.set_ylabel(None)
            ax.set_xlabel(None)
            ax.set_xticks([0, 1], ['More', 'Less'])
            ax.set_xlim(-.5, 1.5)
        axes[row, 0].set_ylabel(cfg.COND_DICT[cond])
    fig.supxlabel('Affected hemisphere')
    fig.supylabel(r'Relative Power [%]')
    plt.tight_layout()
    _save_fig(fig, f'{prefix}reproduce_shreve_relative',
              bbox_inches=None, transparent=True,
              save_dir=join(cfg.FIG_PAPER, fig_dir))


def periodic_bands_by_hemisphere(df_per, fig_dir='Figure_S7', prefix=''):
    df_per = df_per[~df_per.project.isin(['all'])]
    df_per = df_per[df_per.cond.isin(['off']) |
                    ((df_per.cond.isin(['on'])
                      & df_per.dominant_side_consistent))]

    conds = ['off', 'on']
    yvals = ['alpha_beta_fm_mean_log', 'alpha_fm_mean_log',
             'beta_low_fm_mean_log', 'beta_high_fm_mean_log',
             'gamma_fm_mean_log', 'full_fm_band_aperiodic_log']
    n_rows = len(conds)
    n_col = len(yvals)
    severity = ['severe side', 'mild side']

    fig, axes = plt.subplots(n_rows, n_col, figsize=(4.4, 1.5), sharey=False,
                             sharex=True)
    axes = _axes2d(axes, n_rows, n_col)
    for row, cond in enumerate(conds):

        x = f"patient_symptom_dominant_side_BR_{cond}"
        df_plot = df_per[(df_per.cond == cond)]

        for col, y in enumerate(yvals):
            ax = axes[row, col]
            if 'aperiodic' in y:
                kind = 'periodicAP'
            else:
                kind = 'periodic'
            if row == 1:
                kind += '2'
            sns.pointplot(df_plot, y=y, x=x, ax=ax, color=cfg.COLOR_DIC[kind],
                          order=severity, errorbar=('ci', 95))
            _stat_anno(ax, df_plot, x, y, fontsize=FONTSIZE_ASTERISK)

            if row == 0:
                ax.set_title(cfg.PLOT_LABELS_SHORT[y].replace(' mean', ''))
            ax.set_ylabel(None)
            ax.set_xlabel(None)
            ax.set_xticks([0, 1], ['More', 'Less'])
            ax.set_xlim(-.5, 1.5)
        axes[row, 0].set_ylabel(cfg.COND_DICT[cond])
    fig.supxlabel('Affected hemisphere')
    fig.supylabel(r'Abs. Power [[log($\mu$V$^2/Hz$)]')
    plt.tight_layout()
    _save_fig(fig, f'{prefix}reproduce_shreve_periodic',
              bbox_inches=None, transparent=True,
              save_dir=join(cfg.FIG_PAPER, fig_dir))
