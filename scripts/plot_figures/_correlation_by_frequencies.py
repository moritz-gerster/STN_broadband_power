"""Helping functions."""
import warnings
from itertools import product
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

import scripts.config as cfg
from scripts.corr_stats import _get_freqs_correlation_stats
from scripts.plot_figures.settings import *
from scripts.utils import get_correlation_df
from scripts.utils_plot import (_add_band_annotations, _save_fig,
                                convert_pvalue_to_asterisks)

c_abs = cfg.COLOR_DIC['absolute']
c_per = cfg.COLOR_DIC['periodic']
c_ap = cfg.COLOR_DIC['periodicAP']
c_norm = cfg.COLOR_DIC['normalized']
c_insig = 'grey'


def barplot_UPDRS_periodic_ax(ax, df_corrs, palette=None, output_file=None):
    """Add barplot to correlation plot over freqs."""
    band_cols = df_corrs.band_nme.unique()
    band_nmes = [band.replace(' mean', '') for band in band_cols]
    if palette is None:
        if 'Beta' in band_cols:
            palette = ([(sns.color_palette()[0])]
                       + list(sns.color_palette("flare", 2)))
        else:
            palette = list(sns.color_palette("flare", 3))

    # barplot
    plot_kwargs = {'ax': ax,
                   'x': 'band_nme',
                   'hue': 'band_nme',
                   'y': 'rho', "order": band_cols,
                   'legend': False, 'data': df_corrs, 'width': 0.6,
                   'palette': palette}
    sns.barplot(**plot_kwargs)
    ax.axhline(0, color='k', lw=LINEWIDTH_AXES, ls='--')

    # add significance star for bars where pval < 0.05
    bars_pooled = ax.containers
    ymin, ymax = ax.get_ylim()
    yscale = ymax - ymin
    for bar, band in zip(bars_pooled, band_cols):
        df_band = df_corrs[(df_corrs.band_nme == band)]
        pvalue = df_band.pval.values[0]
        text = convert_pvalue_to_asterisks(pvalue, stack_vertically=False)
        x_bar = bar[0].get_x() + bar[0].get_width() / 2
        y_bar = bar[0].get_height()
        offset = max(y_bar, 0) - 0.06 * yscale
        va = 'bottom'
        ax.annotate(text, xy=(x_bar, offset), ha='center', va=va,
                    fontsize=FONTSIZE_ASTERISK, linespacing=.15)
        print(f'{band}: rho={y_bar:.2f}, pval={pvalue:.2f}', file=output_file)

    # set axis
    xticks = np.array(ax.get_xticks())
    ax.set_xticks(xticks, labels=band_nmes)
    ax.set_xlabel(None)

    # output legend
    handles = [Patch(color=color) for color in palette]
    labels = [band_nme for band_nme in band_nmes]
    return handles, labels


def df_corr_freq(df_plot, x, y, average_hemispheres=None, rolling_mean=None,
                 xmax=45, corr_method='spearman', n_perm=N_PERM_CORR,
                 remove_ties=True, projects=cfg.PROJECT_ORDER_SLIM):
    """Get correlation over each frequency bin."""
    if isinstance(rolling_mean, int) and rolling_mean > 1:
        msg = 'Rolling mean incorrectly implemented for pvalues'
        raise NotImplementedError(msg)
    df_corrs = []
    projects = [proj for proj in projects if proj in df_plot.project.unique()]
    if average_hemispheres is None:
        if y == 'UPDRS_III':
            average_hemispheres = True
        elif 'contra' in y:
            average_hemispheres = False
    for project in projects:
        df_proj = df_plot[df_plot.project == project]
        df_corr = _get_freqs_correlation_stats(
            df_proj, x, y, average_hemispheres, xmax, corr_method=corr_method,
            remove_ties=remove_ties, n_perm=n_perm
            )
        df_corrs.append(df_corr)
    df_corrs = pd.concat(df_corrs, ignore_index=True)
    return df_corrs


def plot_psd_updrs_correlation(df_corrs, x, y, kind, fig_dir=None, prefix='',
                               output_file=None):
    projects = [proj for proj in cfg.PROJECT_ORDER_SLIM
                if proj in df_corrs.project.unique()]
    updrs = y.replace('UPDRS_', '').replace('III', 'III_mean')
    line_widths = {proj: .25 for proj in projects}
    line_widths['all'] = 1

    if x in ['psd', 'asd', 'psd_log']:
        x_plot = "psd_freqs"
    elif 'fm' in x:
        x_plot = "fm_freqs"
    y_plot = f"corr_{x}_{y}"
    y_pval = f"pval_{x}_{y}"
    cond = df_corrs.cond.unique()[0]
    n_perm = df_corrs.n_perm.unique()[0]
    xmax = df_corrs[x_plot].max()

    fig, ax = plt.subplots(1, 1, figsize=(2, 1.3))
    for project in projects:
        df_corr = df_corrs[df_corrs.project == project]
        lw = line_widths[project]
        sample_size = df_corr.sample_size.unique()[0]
        if 'contra' not in y:
            sample_size_str = f'{cfg.SAMPLE_PAT}={sample_size}'
        else:
            sample_size_str = f'{cfg.SAMPLE_STN}={sample_size}'
        label = (f"{cfg.COND_DICT[cond]} {sample_size_str}"
                 if project == 'all' else None)
        sns.lineplot(data=df_corr, ax=ax, x=x_plot, lw=lw, y=y_plot,
                     label=label, color=cfg.COLOR_DIC[project + "3"])

        if project == 'all':
            freqs_significant = df_corr[y_pval] < 0.05
            ymin, ymax = ax.get_ylim()
            x_arr = df_corr[x_plot].values.astype(float)
            y_arr = np.ones_like(x_arr) * ymin
            x_arr[~freqs_significant] = np.nan
            y_arr[~freqs_significant] = np.nan
            ax.plot(x_arr, y_arr, color=cfg.COLOR_DIC[project + "3"],
                    lw=1)
            ax.legend(handlelength=1, loc='upper right')
            ax.set_ylim(ymin-.02, ymax)

            # Find indices where the difference between consecutive elements
            # is greater than 1
            x_arr = x_arr[~np.isnan(x_arr)]
            diff = np.diff(x_arr)
            breaks = np.where(diff > 1)[0]
            # Split the array into clusters based on the breaks
            clusters = np.split(x_arr, breaks + 1)
            # Format the clusters into desired range strings
            formatted_clusters = [f"{cluster[0]:.0f}-{cluster[-1]:.0f} Hz"
                                  for cluster in clusters]
            output = ", ".join(formatted_clusters)
            print("Significant clusters:", output, file=output_file)
    ax.set_xlim((0, df_corrs[x_plot].max()))
    ax.hlines(0, *ax.get_xlim(), color='k', lw=LINEWIDTH_AXES, ls='--')
    corr_method = df_corrs.corr_method.unique()[0]
    ylabel = _get_ylabel(corr_method)
    ax.set_ylabel(ylabel)
    ax.set_xticks(XTICKS_FREQ_low)
    ax.set_xticklabels(XTICKS_FREQ_low_labels)
    ax.set_xlabel('Frequency [Hz]')
    plt.tight_layout()
    if fig_dir:
        _save_fig(fig, f'{fig_dir}/{prefix}psd_UPDRS_correlation_{corr_method}_'
                  f'{kind}_{cond}_{updrs}_{xmax}Hz_nperm={n_perm}',
                  cfg.FIG_PAPER, bbox_inches=None);


def _get_ylabel(corr_method):
    if corr_method == 'spearman':
        ylabel = r'$\rho$'
    elif corr_method == 'within':
        ylabel = r"$r_{rm}$"
    elif corr_method == 'withinRank':
        ylabel = r"$r_{\text{rank rm}}$"
    return ylabel


def plot_psd_updrs_correlation_and_bar(df_corrs, df_corrs_bar,
                                       save_dir=None, figsize=(2.6, 1.5),
                                       legend=False, palette_barplot=None,
                                       band_annos=None, xmin=2,
                                       info_title=False, prefix='',
                                       output_file=None,
                                       ylim=None, fill_significance=False):
    assert df_corrs.project.nunique() == 1
    kinds = '_'.join(kind for kind in df_corrs.kind.unique())
    x = df_corrs['x'].unique()[0]
    y = df_corrs['y'].unique()[0]
    updrs = y.replace('UPDRS_', '').replace('III', 'III_mean')

    if x in ['psd', 'asd', 'psd_log']:
        x_plot = "psd_freqs"
    elif 'fm' in x:
        x_plot = "fm_freqs"
    y_plot = f"corr_{x}_{y}"
    y_pval = f"pval_{x}_{y}"
    conds = df_corrs.cond.unique()
    kinds = df_corrs.kind.unique()
    n_perm = df_corrs.n_perm.unique()[0]
    xmax = df_corrs[x_plot].max()
    if df_corrs_bar is None:
        fig, axes = plt.subplots(1, 1, figsize=figsize, sharey=True)
        # plot correlation over freqs
        ax = axes
    else:
        corr_method = df_corrs.corr_method.unique()[0]
        if corr_method == 'spearman':
            width_ratios = [1, 0.4]
        elif corr_method.startswith('within'):
            width_ratios = [1, 0.2 * len(df_corrs_bar.band_nme.unique())]
        fig, axes = plt.subplots(1, 2, figsize=figsize,
                                 width_ratios=width_ratios, sharey=True)
        # plot correlation over freqs
        ax = axes[0]
    for cond in conds:
        for kind in kinds:
            df_corr = df_corrs[(df_corrs.kind == kind)
                               & (df_corrs.cond == cond)]
            label = cfg.COND_DICT[cond]
            c = '2' if cond == 'on' else ''
            color = cfg.COLOR_DIC[kind + c]
            sns.lineplot(data=df_corr, ax=ax, x=x_plot, lw=1, y=y_plot,
                         label=label, color=color)
            freqs_significant = df_corr[y_pval] < 0.05

            sample_size = df_corr['sample_size'].unique()[0]
            if df_corr['hemispheres_averaged'].unique()[0]:
                sampling_str = 'sub'
            else:
                sampling_str = 'hemi'
            sample_size_str = (f'{kind} {cond} '
                               rf'$n_{{{sampling_str}}}={sample_size}$')
            print(sample_size_str, file=output_file)

            # Find indices where the difference between consecutive elements
            # is greater than 1
            x_arr = df_corr[x_plot].values.astype(float)
            x_arr[~freqs_significant] = np.nan
            x_arr = x_arr[~np.isnan(x_arr)]
            diff = np.diff(x_arr)
            breaks = np.where(diff > 1)[0]
            # Split the array into clusters based on the breaks
            clusters = np.split(x_arr, breaks + 1)
            # Format the clusters into desired range strings
            formatted_clusters = [f"{cluster[0]:.0f}-{cluster[-1]:.0f} Hz"
                                  for cluster in clusters]
            output = ", ".join(formatted_clusters)
            print("Significant clusters:", output, file=output_file)
            if fill_significance:
                if freqs_significant.sum() > 0:
                    ax.fill_between(df_corr[x_plot], 0, df_corr[y_plot],
                                    color=color, alpha=0.2,
                                    where=freqs_significant, label=None)

    # set axis
    ax.axhline(0, color='k', lw=LINEWIDTH_AXES, ls='--')
    corr_method = df_corrs.corr_method.unique()[0]
    if corr_method == 'spearman':
        ylabel = r"$\rho$"
    elif corr_method == 'within':
        ylabel = r"$r_{rm}$"
    elif corr_method == 'withinRank':
        ylabel = r"$r_{\text{rank rm}}$"
    ax.set_ylabel(ylabel)
    alpha = 1 if cond == 'on' else 0
    ax.set_xlabel('Frequency [Hz]', alpha=alpha)
    handles, labels = ax.get_legend_handles_labels()

    # bar plot
    asymmetric_subjects = df_corrs.asymmetric_subjects.unique()[0]
    if asymmetric_subjects == True:
        consistent_str = '_consistent'
    elif asymmetric_subjects == False:
        consistent_str = '_inconsistent'
    elif asymmetric_subjects is None:
        consistent_str = ''
    log = '_log' if x.endswith('_log') else ''
    if df_corrs_bar is not None:
        bar_str = 'Hz+bar'
        barplot_UPDRS_periodic_ax(axes[1], df_corrs_bar,
                                  palette=palette_barplot,
                                  output_file=output_file)
    else:
        bar_str = ''
    kind_str = '_'.join(kinds)
    fname = (f'{prefix}psd{log}_UPDRS_correlation_{corr_method}_'
             f'{kind_str}_{cond}_{updrs}_{xmax}Hz{bar_str}'
             f'{consistent_str}_nperm={n_perm}')

    if legend:
        ax.legend()
    else:
        ax.legend().remove()

    # horizontal lines for significance - do at the end for correct axis limits
    if ylim is not None:
        ax.set_ylim(ylim)
    ymin, ymax = ax.get_ylim()
    yscale = ymax - ymin
    ydiff = 0.025 * yscale
    offset = ymin - ydiff/2
    for cond in conds:
        for kind in reversed(kinds):
            df_corr = df_corrs[(df_corrs.kind == kind)
                               & (df_corrs.cond == cond)]
            c = '2' if cond == 'on' else ''
            color = cfg.COLOR_DIC[kind + c]
            freqs_significant = df_corr[y_pval] < 0.05
            if freqs_significant.sum() > 0:
                # Add horizontal lines for significance
                significant_freqs = df_corr[x_plot][freqs_significant].values
                splits = np.where(np.diff(significant_freqs) != 1)[0] + 1
                clusters = np.split(significant_freqs, splits)
                for cluster in clusters:
                    ax.plot([cluster[0], cluster[-1]],
                            [offset, offset], color=color, lw=1)
                offset += ydiff
    ax.set_ylim(ymin - ydiff*1.5, ymax)
    if xmax == 45:
        if df_corrs_bar is None:
            xticks = XTICKS_FREQ_low + [60]
            labels = XTICKS_FREQ_low_labels + [60]
        else:
            # skip last xtick at 100 Hz due to barplot xticks
            xticks = XTICKS_FREQ_low[:-1]
            labels = XTICKS_FREQ_low_labels[:-1]
    elif xmax == 60:
        if df_corrs_bar is None:
            xticks = XTICKS_FREQ_high
            labels = XTICKS_FREQ_high
        else:
            # skip last xtick at 100 Hz due to barplot xticks
            xticks = XTICKS_FREQ_high
            labels = XTICKS_FREQ_high
            labels[-2] = ''  # remove 60 Hz
    else:
        if df_corrs_bar is None:
            xticks = XTICKS_FREQ_high
            labels = XTICKS_FREQ_high
        else:
            xticks = XTICKS_FREQ_high[:-1]
            labels = XTICKS_FREQ_high[:-1]
    ax.set_xticks(xticks, labels=labels)
    ax.set_xlim((xmin, df_corrs[x_plot].max()))

    _add_band_annotations(band_annos, ax, short=False, y=1.07,
                          invisible=False)  # do after setting xlim
    if info_title:
        kind_str = cfg.KIND_DICT[kind]
        cond_str = cfg.COND_DICT[cond]
        updrs_str = cfg.PLOT_LABELS[y]
        ax.set_title(f'{kind_str}, {cond_str}, {updrs_str}, {sample_size_str}')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0025)
    if save_dir:
        _save_fig(fig, join(save_dir, fname), cfg.FIG_PAPER,
                  transparent=False,
                  bbox_inches=None)


def figure2(dataframes, kinds=['normalized', 'absolute', 'periodic'],
            conds=['offon_abs'], n_perm=N_PERM_CORR, xmax=45,
            asymmetric_subjects=None, fig_dir=None, prefix='',
            output_file=None,
            y="UPDRS_III", ylim=None,
            corr_method='spearman'):
    df_corr_bar = None
    for cond in conds:
        df_corrs_all = []
        for kind in kinds:
            if kind == 'normalized':
                df_plot = dataframes['df_norm']
                x = 'psd_log'
                # xmax = 45
            elif kind == 'normalizedInce':
                df_plot = dataframes['df_normInce']
                x = 'psd_log'
            elif kind  == 'absolute':
                df_plot = dataframes['df_abs']

                x = 'psd_log'
                palette_barplot = [c_abs, c_abs, c_abs]
                band_annos = BANDS
                band_cols = ['theta_abs_mean_log',
                             'beta_low_abs_mean_log',
                             'gamma_low_abs_mean_log']
                data = df_plot[(df_plot.cond == cond)
                               & (df_plot.project == 'all')]
                df_corr_bar = get_correlation_df(data, y, total_power=True,
                                                 use_peak_power=True,
                                                 n_perm=n_perm,
                                                 band_cols=band_cols,
                                                 corr_method=corr_method)
                df_corr_bar['kind'] = kind
            elif kind  == 'periodic':
                df_plot = dataframes['df_per']
                x = 'fm_psd_peak_fit_log'
            elif kind  == 'periodicAP':
                df_plot = dataframes['df_per']
                x = 'fm_psd_peak_fit_log'
                data = df_plot[(df_plot.cond == cond)
                               & (df_plot.project == 'all')]
                df_corr_bar = get_correlation_df(data, y, total_power=False,
                                                 use_peak_power=True,
                                                 n_perm=n_perm,
                                                 output_file=output_file,
                                                 bands=[],
                                                 corr_method=corr_method)
                df_corr_bar['kind'] = kind
            elif kind  == 'periodicFULL':
                df_plot = dataframes['df_per']
                x = 'fm_fooofed_spectrum_log'
                xmax = 40  # max for fm Litvak

            if kind.startswith('periodic'):
                data = df_plot[(df_plot.cond == cond)
                               & (df_plot.project == 'all')]
                palette_barplot = [c_ap, c_per, c_per]
                band_annos = BANDS
                band_cols = ['fm_offset_log',
                             'beta_low_fm_mean_log',
                             'gamma_low_fm_mean_log',
                             ]
                df_corr_bar = get_correlation_df(data, y, total_power=False,
                                                 use_peak_power=True,
                                                 n_perm=n_perm,
                                                 bands=None,
                                                 band_cols=band_cols,
                                                 output_file=output_file,
                                                 corr_method=corr_method)
                df_corr_bar['kind'] = kind

            data = df_plot[(df_plot.cond == cond) & (df_plot.project == 'all')]
            df_corrs = df_corr_freq(data, x, y, corr_method=corr_method,
                                    n_perm=n_perm, xmax=xmax)
            df_corrs['kind'] = kind
            df_corrs['n_perm'] = n_perm
            df_corrs['asymmetric_subjects'] = asymmetric_subjects
            df_corrs_all.append(df_corrs)
        df_corrs_all = pd.concat(df_corrs_all)
        plot_psd_updrs_correlation_and_bar(df_corrs_all, df_corr_bar,
                                           save_dir=fig_dir,
                                           prefix=prefix,
                                           legend=False, ylim=ylim,
                                           palette_barplot=palette_barplot,
                                           output_file=output_file,
                                           band_annos=band_annos,
                                           figsize=(3.5, 1.2))

rename_periodic = {'fm_freqs': 'psd_freqs',
                   'fm_psd_peak_fit': 'psd',
                   'fm_psd_peak_fit_log': 'psd_log',
                   'fm_psd_ap_fit': 'psd',
                   'fm_psd_ap_fit_log': 'psd_log',
                   'fm_fooofed_spectrum': 'psd',
                   'fm_fooofed_spectrum_log': 'psd_log',
                   'psd': 'psd',
                   'psd_log': 'psd_log',
                   'corr_fm_psd_peak_fit_UPDRS_bradyrigid_contra': 'corr_psd_UPDRS_bradyrigid_contra',
                   'pval_fm_psd_peak_fit_UPDRS_bradyrigid_contra': 'pval_psd_UPDRS_bradyrigid_contra',
                   'corr_fm_psd_peak_fit_log_UPDRS_bradyrigid_contra': 'corr_psd_log_UPDRS_bradyrigid_contra',
                   'pval_fm_psd_peak_fit_log_UPDRS_bradyrigid_contra': 'pval_psd_log_UPDRS_bradyrigid_contra',
                   'corr_fm_psd_ap_fit_UPDRS_bradyrigid_contra': 'corr_psd_UPDRS_bradyrigid_contra',
                   'pval_fm_psd_ap_fit_UPDRS_bradyrigid_contra': 'pval_psd_UPDRS_bradyrigid_contra',
                   'corr_fm_psd_ap_fit_log_UPDRS_bradyrigid_contra': 'corr_psd_log_UPDRS_bradyrigid_contra',
                   'pval_fm_psd_ap_fit_log_UPDRS_bradyrigid_contra': 'pval_psd_log_UPDRS_bradyrigid_contra',
                   'pval_fm_psd_peak_fit_log_UPDRS_bradyrigid_contra': 'pval_psd_log_UPDRS_bradyrigid_contra',
                   'corr_fm_fooofed_spectrum_UPDRS_bradyrigid_contra': 'corr_psd_UPDRS_bradyrigid_contra',
                   'pval_fm_fooofed_spectrum_UPDRS_bradyrigid_contra': 'pval_psd_UPDRS_bradyrigid_contra',
                   'corr_fm_fooofed_spectrum_log_UPDRS_bradyrigid_contra': 'corr_psd_log_UPDRS_bradyrigid_contra',
                   'pval_fm_fooofed_spectrum_log_UPDRS_bradyrigid_contra': 'pval_psd_log_UPDRS_bradyrigid_contra',
                   'corr_fm_psd_peak_fit_UPDRS_III': 'corr_psd_UPDRS_III',
                   'pval_fm_psd_peak_fit_UPDRS_III': 'pval_psd_UPDRS_III',
                   'corr_fm_psd_ap_fit_UPDRS_III': 'corr_psd_UPDRS_III',
                   'pval_fm_psd_ap_fit_UPDRS_III': 'pval_psd_UPDRS_III',
                   'corr_fm_psd_ap_fit_log_UPDRS_III': 'corr_psd_log_UPDRS_III',
                   'pval_fm_psd_ap_fit_log_UPDRS_III': 'pval_psd_log_UPDRS_III',
                   'corr_fm_psd_peak_fit_log_UPDRS_III': 'corr_psd_log_UPDRS_III',
                   'pval_fm_psd_peak_fit_log_UPDRS_III': 'pval_psd_log_UPDRS_III',
                   'pval_fm_psd_peak_fit_log_UPDRS_III': 'pval_psd_log_UPDRS_III',
                   'corr_fm_fooofed_spectrum_UPDRS_III': 'corr_psd_UPDRS_III',
                   'pval_fm_fooofed_spectrum_UPDRS_III': 'pval_psd_UPDRS_III',
                   'corr_fm_fooofed_spectrum_log_UPDRS_III': 'corr_psd_log_UPDRS_III',
                   'pval_fm_fooofed_spectrum_log_UPDRS_III': 'pval_psd_log_UPDRS_III',
                   }


def plot_psd_updrs_correlation_multi(df_corrs, fig_dir=None, figsize=(7, 1.3),
                                     xlabel=None,
                                     legend=False,  band_annos=None, xmin=2,
                                     info_title=False, prefix='',
                                     ylim=None):
    assert df_corrs.project.nunique() == 1
    x = df_corrs.x.unique()[0]
    y = df_corrs.y.unique()[0]
    kinds = df_corrs.kind.unique()
    updrs = y.replace('UPDRS_', '').replace('III', 'III_mean')
    kinds_str = '_'.join(kind for kind in df_corrs.kind.unique())

    if x in ['psd', 'asd', 'psd_log']:
        x_plot = "psd_freqs"
    elif 'fm' in x:
        x_plot = "fm_freqs"
    y_plot = f"corr_{x}_{y}"
    y_pval = f"pval_{x}_{y}"
    conds = df_corrs.cond.unique()
    cond_str = '_'.join(cond for cond in conds)
    n_perm = df_corrs.n_perm.unique()[0]
    xmax = df_corrs[x_plot].max()
    corr_method = df_corrs.corr_method.unique()[0]

    kinds_conds = list(product(kinds, conds))
    n_cols = len(kinds_conds)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=True)
    for idx, (kind, cond) in enumerate(kinds_conds):
        ax = axes[idx]
        df_plot = df_corrs[(df_corrs.kind == kind) & (df_corrs.cond == cond)]
        label = cfg.COND_DICT[cond]
        c = '2' if cond == 'on' else ''
        color = cfg.COLOR_DIC[kind + c]
        sns.lineplot(data=df_plot, ax=ax, x=x_plot, lw=1, y=y_plot,
                    label=label, color=color)

        # horizontal lines for significance - do at the end for correct axis
        # limits
        if ylim is not None:
            ax.set_ylim(ylim)
        ymin, ymax = ax.get_ylim()
        yscale = ymax - ymin
        ydiff = 0.025 * yscale
        offset = ymin - ydiff/2
        freqs_significant = df_plot[y_pval] < 0.05
        if freqs_significant.sum() > 0:
            # Add horizontal lines for significance
            significant_freqs = df_plot[x_plot][freqs_significant]
            splits = np.where(np.diff(significant_freqs) != 1)[0] + 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                clusters = np.split(significant_freqs, splits)
            for cluster in clusters:
                ax.plot([cluster.iloc[0], cluster.iloc[-1]],
                        [offset, offset], color=color, lw=1)
            offset += ydiff
        ax.set_ylim(ymin - ydiff*1.5, ymax)

        # set axis
        ax.axhline(0, color='k', lw=LINEWIDTH_AXES, ls='--')
        corr_method = df_corrs.corr_method.unique()[0]
        if corr_method == 'spearman':
            ylabel = r"Spearman $\rho$ "
        elif corr_method == 'within':
            ylabel = r"$r_{rm}$"
        elif corr_method == 'withinRank':
            ylabel = r"$r_{\text{rank rm}}$"
        # labelpad important to match raster with cluster stats
        labelpad = 0 if kind == 'normalized' else 2.5
        ax.set_ylabel(ylabel, labelpad=labelpad)
        ax.set_xlabel(None)

        if legend:
            ax.legend()
        else:
            ax.legend().remove()

        if xmax == 45:
            xticks = XTICKS_FREQ_low + [60]
            labels = XTICKS_FREQ_low_labels + [60]
        elif xmax == 60:
            xticks = XTICKS_FREQ_high
            labels = XTICKS_FREQ_high_labels_skip13
        else:
            # skip last xtick at 100 Hz due to barplot xticks
            xticks = XTICKS_FREQ_high[:-1]
            labels = XTICKS_FREQ_high_labels_skip13[:-1]
        ax.set_xticks(xticks, labels=labels)
        ax.set_xlim((xmin, df_corrs[x_plot].max()))

        # invisible = True if kind != 'absolute' else False
        # short = False if kind == 'absolute' else True
        if band_annos:
            _add_band_annotations(band_annos[kind], ax, short=False, y=1.07,
                                  invisible=False)  # do after setting xlim
        if info_title:
            sample_size = df_corrs['sample_size'].unique()[0]
            if df_corrs['hemispheres_averaged'].unique()[0]:
                sampling_str = 'sub'
            else:
                sampling_str = 'hemi'
            sample_size_str = rf'$n_{{{sampling_str}}}={sample_size}$'
            kind_str = cfg.KIND_DICT[kind]
            cond_str = cfg.COND_DICT[cond]
            updrs_str = cfg.PLOT_LABELS[y]
            ax.set_title(f'{kind_str}, {cond_str}, {updrs_str}, '
                         f'{sample_size_str}')
    if xlabel:
        fig.supxlabel(xlabel)
    log = '_log' if x.endswith('_log') else ''

    fname = (f'{prefix}psd{log}_UPDRS_correlation_{corr_method}_'
             f'{kinds_str}_{cond_str}_{updrs}_{xmax}Hz'
             f'_nperm={n_perm}')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    if fig_dir:
        _save_fig(fig, join(fig_dir, fname), cfg.FIG_PAPER,
                  transparent=False, bbox_inches=None)


def get_corrs_kinds(dataframes, kinds, conds, n_perm=N_PERM_CORR, xmax=60,
                    scale='log', corr_method='withinRank', remove_ties=True,
                    y="UPDRS_bradyrigid_contra"):
    df_corrs_all = []
    for kind in kinds:
        for cond in conds:
            if kind == 'normalized':
                df = dataframes['df_norm']
            elif kind == 'normalizedInce':
                df = dataframes['df_normInce']
            elif kind in ['absolute', 'periodic', 'periodicAP', 'periodicFULL']:
                df = dataframes['df_abs']
            df = df[(df.cond == cond) & (df.project == 'all')]

            # # remove subject where asymmetry switches between On and Off
            if cond == 'on':
                # only include consistent asymmetry for ON subjects to
                # exclude possible LDOPA side effects
                df = df[df.dominant_side_consistent]
            if kind == 'normalized':
                x = 'psd'
            elif kind == 'normalizedInce':
                x = 'psd'
            elif kind  == 'absolute':
                x = 'psd'
            elif kind  == 'periodic':
                x = 'fm_psd_peak_fit'
            elif kind  == 'periodicAP':
                x = 'fm_psd_ap_fit'
            elif kind  == 'periodicFULL':
                x = 'fm_fooofed_spectrum'

            if 'offon' in cond:
                assert x.endswith('_log')
            else:
                if scale == 'log':
                    x += '_log'
                elif scale == 'linear':
                    assert not x.endswith('log')
                else:
                    raise ValueError(f'Unknown scale: {scale}')

            df_corrs = df_corr_freq(df, x, y, corr_method=corr_method,
                                    remove_ties=remove_ties,
                                    n_perm=n_perm, xmax=xmax)
            df_corrs['kind'] = kind
            df_corrs['scale'] = scale
            df_corrs['n_perm'] = n_perm
            df_corrs['cond'] = cond
            df_corrs.rename(columns=rename_periodic, inplace=True)
            df_corrs_all.append(df_corrs)
    df_corrs_all = pd.concat(df_corrs_all)
    x = rename_periodic[x]
    df_corrs_all['x'] = x
    return df_corrs_all