from os.path import join

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

import scripts.config as cfg
from scripts.plot_figures.settings import *
from scripts.utils_plot import _add_band_annotations, _save_fig


def exemplary_broadband_shifts(df_per, fig_dir=None, prefix=''):
    xmin_lin = 2
    xmax = 60
    bands = ['delta', 'theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

    cond = 'on'
    col = f"patient_symptom_dominant_side_BR_{cond}"

    n_cols = len(cfg.EXEMPLARY_SUBS_APERIODIC)
    fig, axes = plt.subplots(1, n_cols, figsize=(7, 1.5), sharey=True)

    for i, sub in enumerate(cfg.EXEMPLARY_SUBS_APERIODIC):

        df_sub = df_per[(df_per.subject == sub) & (df_per.cond == cond)]
        sub_nme = df_sub.subject_nme.values[0]

        df_mild_sub = df_sub[df_sub[col] == 'mild side']
        df_severe_sub = df_sub[df_sub[col] == 'severe side']

        freqs = df_mild_sub.psd_freqs.values[0]
        freqs_fm = df_mild_sub.fm_freqs.values[0]

        mask_fm = (freqs_fm >= xmin_lin) & (freqs_fm <= xmax)
        mask = (freqs >= xmin_lin) & (freqs <= xmax)

        freqs = freqs[mask]
        freqs_fm = freqs_fm[mask_fm]

        psd_mild = df_mild_sub.psd.values[0][mask]
        psd_severe = df_severe_sub.psd.values[0][mask]

        fm_mild_total = df_mild_sub.fm_fooofed_spectrum.values[0][mask_fm]
        fm_mild_ap = df_mild_sub.fm_psd_ap_fit.values[0][mask_fm]

        fm_severe_total = df_severe_sub.fm_fooofed_spectrum.values[0][mask_fm]
        fm_severe_ap = df_severe_sub.fm_psd_ap_fit.values[0][mask_fm]

        mild_score = df_mild_sub.UPDRS_bradyrigid_contra.values[0]
        severe_score = df_severe_sub.UPDRS_bradyrigid_contra.values[0]

        # Extract differences
        ax = axes[i]
        ax.semilogy(freqs, psd_severe, 'k',
                    label=f'Severe (BR={severe_score:.0f})')
        ax.semilogy(freqs, psd_mild, 'k--',
                    label=f'Mild (BR={mild_score:.0f})')
        if i == 0:
            label_per = 'Periodic'
            label_ap = 'Aperiodic'
            label_gamma = 'Periodic $\\gamma$'
            bbox_to_anchor = (1, 1)
        else:
            label_per = None
            label_ap = None
            label_gamma = None
            bbox_to_anchor = (1, 1)
        loc = 'upper right'
        sub_handle = mlines.Line2D([], [], color=cfg.COLORS_SPECIAL_SUBS[i],
                                   marker=cfg.SYMBOLS_SPECIAL_SUBS[i],
                                   markersize=2, lw=0
                                   )
        handles, labels = ax.get_legend_handles_labels()
        handles = [sub_handle] + handles
        labels = [sub_nme] + labels
        leg1 = ax.legend(handles, labels, handlelength=1, loc=loc,
                         borderaxespad=.2,
                         bbox_to_anchor=bbox_to_anchor)
        ax.semilogy(freqs_fm, fm_severe_total,
                    cfg.COLOR_DIC['periodic'], label=label_per)
        ax.semilogy(freqs_fm, fm_severe_ap,
                    cfg.COLOR_DIC['periodicAP'], label=label_ap)
        ax.semilogy(freqs_fm, fm_mild_total,
                    cfg.COLOR_DIC['periodic'], ls='--', label=None)
        ax.semilogy(freqs_fm, fm_mild_ap,
                    cfg.COLOR_DIC['periodicAP'], ls='--', label=None)

        # fill periodic gamma for subbands
        band = 'gamma'
        f_low, f_high = cfg.BANDS[band]
        color = cfg.BAND_COLORS['gamma_low']
        ax.fill_between(freqs_fm, fm_severe_total, fm_severe_ap,
                        where=(freqs_fm >= f_low) & (freqs_fm <= f_high),
                        color=color, alpha=1, label=label_gamma)
        ax.fill_between(freqs_fm, fm_mild_total, fm_mild_ap,
                        where=(freqs_fm >= f_low) & (freqs_fm <= f_high),
                        color=color, alpha=.5)
        xticks = XTICKS_FREQ_high
        xticklabels = XTICKS_FREQ_high_labels_skip9
        if i != len(cfg.EXEMPLARY_SUBS_APERIODIC) - 1:
            xticks = xticks[:-2]
            xticklabels = xticklabels[:-2]
        ax.set_xticks(xticks, labels=xticklabels)
        ax.set_xlim([xmin_lin, xmax])
        mask = freqs > xmin_lin
        _add_band_annotations(bands, ax, short=True, y=1.08)

        handles, labels = ax.get_legend_handles_labels()
        handles, labels = handles[2:], labels[2:]
        ax.add_artist(leg1)
        if i == 0:
            leg2 = ax.legend(handles, labels,
                             bbox_to_anchor=(0, 0), loc='lower left',
                             frameon=False, borderaxespad=0, handlelength=1)
            ax.add_artist(leg2)
        xlabel = 'Frequency [Hz]' if i == 2 else None
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(r'PSD [$\mu V^2/Hz$]')
    plt.tight_layout()
    _save_fig(fig, f'{prefix}all_subs_ap_fits_{cond}',
              bbox_inches=None, save_dir=join(cfg.FIG_PAPER, fig_dir))


def exemplary_gamma(df_per, fig_dir=None, prefix=''):
    xmin_lin = 2
    xmax = 60
    bands = ['delta', 'theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

    cond = 'on'
    col = f"patient_symptom_dominant_side_BR_{cond}"

    n_cols = len(cfg.EXEMPLARY_SUBS_GAMMA)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.2, 1.5), sharey=True)

    for i, sub in enumerate(cfg.EXEMPLARY_SUBS_GAMMA):

        df_sub = df_per[(df_per.subject == sub) & (df_per.cond == cond)]
        sub_nme = df_sub.subject_nme.values[0]

        df_mild_sub = df_sub[df_sub[col] == 'mild side']
        df_severe_sub = df_sub[df_sub[col] == 'severe side']

        freqs = df_mild_sub.psd_freqs.values[0]
        freqs_fm = df_mild_sub.fm_freqs.values[0]

        mask_fm = (freqs_fm >= xmin_lin) & (freqs_fm <= xmax)
        mask = (freqs >= xmin_lin) & (freqs <= xmax)

        freqs = freqs[mask]
        freqs_fm = freqs_fm[mask_fm]

        psd_mild = df_mild_sub.psd.values[0][mask]
        psd_severe = df_severe_sub.psd.values[0][mask]

        fm_mild_total = df_mild_sub.fm_fooofed_spectrum.values[0][mask_fm]
        fm_mild_ap = df_mild_sub.fm_psd_ap_fit.values[0][mask_fm]

        fm_severe_total = df_severe_sub.fm_fooofed_spectrum.values[0][mask_fm]
        fm_severe_ap = df_severe_sub.fm_psd_ap_fit.values[0][mask_fm]

        mild_score = df_mild_sub.UPDRS_bradyrigid_contra.values[0]
        severe_score = df_severe_sub.UPDRS_bradyrigid_contra.values[0]

        # Extract differences
        ax = axes[i]
        ax.semilogy(freqs, psd_severe, 'k',
                    label=f'Severe (BR={severe_score:.0f})')
        ax.semilogy(freqs, psd_mild, 'k--',
                    label=f'Mild (BR={mild_score:.0f})')
        if i == 0:
            label_per = 'Periodic'
            label_ap = 'Aperiodic'
            label_gamma = 'Periodic $\\gamma$'
            bbox_to_anchor = (1, 1)
        else:
            label_per = None
            label_ap = None
            label_gamma = None
            bbox_to_anchor = (1, 1)
        loc = 'upper right'
        sub_handle = mlines.Line2D([], [], color=cfg.COLORS_SPECIAL_SUBS[i],
                                   marker=cfg.SYMBOLS_SPECIAL_SUBS[i],
                                   markersize=2, lw=0
                                   )
        handles, labels = ax.get_legend_handles_labels()
        handles = [sub_handle] + handles
        labels = [sub_nme] + labels
        leg1 = ax.legend(handles, labels, handlelength=1, loc=loc,
                         borderaxespad=.2,
                         bbox_to_anchor=bbox_to_anchor)
        ax.semilogy(freqs_fm, fm_severe_total,
                    cfg.COLOR_DIC['periodic'], label=label_per)
        ax.semilogy(freqs_fm, fm_severe_ap,
                    cfg.COLOR_DIC['periodicAP'], label=label_ap)
        ax.semilogy(freqs_fm, fm_mild_total,
                    cfg.COLOR_DIC['periodic'], ls='--', label=None)
        ax.semilogy(freqs_fm, fm_mild_ap,
                    cfg.COLOR_DIC['periodicAP'], ls='--', label=None)

        # fill periodic gamma for subbands
        band = 'gamma'
        f_low, f_high = cfg.BANDS[band]
        color = cfg.BAND_COLORS['gamma_low']
        ax.fill_between(freqs_fm, fm_severe_total, fm_severe_ap,
                        where=(freqs_fm >= f_low) & (freqs_fm <= f_high),
                        color=color, alpha=1, label=label_gamma)
        ax.fill_between(freqs_fm, fm_mild_total, fm_mild_ap,
                        where=(freqs_fm >= f_low) & (freqs_fm <= f_high),
                        color=color, alpha=.5)
        xticks = XTICKS_FREQ_high
        xticklabels = XTICKS_FREQ_high_labels_skip9
        if i != len(cfg.EXEMPLARY_SUBS_GAMMA) - 1:
            xticks = xticks[:-2]
            xticklabels = xticklabels[:-2]
        ax.set_xticks(xticks, labels=xticklabels)
        ax.set_xlim([xmin_lin, xmax])
        mask = freqs > xmin_lin
        _add_band_annotations(bands, ax, short=True, y=1.08)

        handles, labels = ax.get_legend_handles_labels()
        handles, labels = handles[2:], labels[2:]
        ax.add_artist(leg1)
        if i == 0:
            leg2 = ax.legend(handles, labels,
                             bbox_to_anchor=(0, 0), loc='lower left',
                             frameon=False, borderaxespad=0, handlelength=1)
            ax.add_artist(leg2)
        ax.set_xlabel(None)
    fig.supxlabel('Frequency [Hz]', y=0.05)
    axes[0].set_ylabel(r'PSD [$\mu V^2/Hz$]')
    plt.tight_layout()
    _save_fig(fig, f'{prefix}all_subs_gamma_fits_{cond}',
              bbox_inches=None, save_dir=join(cfg.FIG_PAPER, fig_dir))
