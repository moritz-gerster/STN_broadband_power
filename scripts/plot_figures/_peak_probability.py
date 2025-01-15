from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

import scripts.config as cfg
from scripts.plot_figures.settings import *
from scripts.utils_plot import _save_fig


def get_peak_probability(df_per, bands=BANDS+['gamma_mid'], project='all',
                         hue='cond'):
    plot_dic = cfg.PLOT_LABELS_SHORT
    band_dic = cfg.BAND_NAMES_GREEK_SHORT

    df = df_per[(df_per.project == 'all')]
    # exclude possible LDOPA side effects
    df = df[(df.cond == 'off') |
            (df.dominant_side_consistent & (df.cond == 'on'))]
    # remove subjects without fooof fit
    subs_no_fit = df[df.fm_has_model.isna()].subject.unique()
    df = df[~df.subject.isin(subs_no_fit)]

    band_cols = [f'{band}_fm_powers_max_log' for band in bands]
    band_nmes = [band_dic[band] for band in bands]
    band_cols += ['fm_exponent', 'fm_offset_log']
    band_nmes += ['1/f', plot_dic['fm_offset_log']]

    msg = f'{set(band_cols) - set(band_nmes)}'
    assert len(band_cols) == len(band_nmes), msg

    if hue == 'cond':
        hue_order = ['off', 'on']
    elif hue == 'severity':
        hue = f"patient_symptom_dominant_side_BR_{cond}"

    df_peaks = []
    for cond in hue_order:
        df_hue = df[df[hue] == cond]
        color = (cfg.COLOR_DIC['periodic'] if cond == 'off'
                 else cfg.COLOR_DIC['periodic2'])
        # remove subject with only one condition
        subject_counts = df_hue.subject.value_counts()
        valid_subjects = subject_counts[subject_counts == 2].index
        df_hue = df_hue[df_hue.subject.isin(valid_subjects)]
        for i, x in enumerate(band_cols):
            sample_size = df_hue[x].notna().sum()
            if 'power' in x:
                n_peaks = (df_hue[x] > 0).sum()
            elif x in ['fm_exponent', 'fm_offset_log']:
                # Aperiodic values can be negative
                n_peaks = (df_hue[x] != 0).sum()
                color = (cfg.COLOR_DIC['periodicAP'] if cond == 'off'
                         else cfg.COLOR_DIC['periodicAP2'])
            probability = n_peaks / sample_size * 100
            dic = {'project': project, 'probability': probability,
                   'color': color, 'band_col': x, 'band_nme': band_nmes[i],
                   'sample_size': sample_size, 'cond': cond}
            df_peaks.append(dic)
    df_peaks = pd.DataFrame(df_peaks)
    return df_peaks


def barplot_peaks(df_peaks, ylim=(0, 100), fig_dir='Figure8', prefix='B__'):
    hue_order = [cond for cond in cfg.COND_ORDER
                 if cond in df_peaks.cond.unique()]
    colors_off = df_peaks[df_peaks.cond == 'off'].color.to_list()
    colors_on = df_peaks[df_peaks.cond == 'on'].color.to_list()

    fig, ax = plt.subplots(1, 1, figsize=(2.3, 1.5), sharey=True)

    sns.barplot(df_peaks, ax=ax, y='probability', x='band_nme', hue='cond',
                hue_order=hue_order, legend=True)
    for bars, colors in zip(ax.containers, (colors_off, colors_on)):
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
    handles = [Patch(facecolor=cfg.COLOR_DIC['all']),
               Patch(facecolor=cfg.COLOR_DIC['all2'])]
    handles = [Patch(facecolor=cfg.COLOR_DIC['periodic']),
               Patch(facecolor=cfg.COLOR_DIC['periodic2'])]
    ax.legend(handles, hue_order)
    ax.set_ylim(ylim)
    ax.set_xlabel('Space label', alpha=0)
    ax.set_ylabel('Fit probability [%]')
    fname = f'{fig_dir}/{prefix}peak_probability_consistent_subs'
    plt.tight_layout()
    _save_fig(fig, fname, cfg.FIG_PAPER,
              transparent=True, bbox_inches=None)

    output_file_path = join(cfg.FIG_PAPER, fig_dir, f"{prefix}_output.txt")
    with open(output_file_path, "w") as output_file:
        for cond in hue_order:
            print(f'\n{cond}: ', file=output_file)
            for band_col in df_peaks.band_col.unique():
                df_sub = df_peaks[(df_peaks.band_col == band_col)
                                  & (df_peaks.cond == cond)]
                band_nme = df_sub.band_nme.values[0]
                probability = df_sub.probability.values[0]
                print(f'{band_nme}: {probability:.1f}%', file=output_file)
