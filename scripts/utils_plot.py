"""Helping plotting functions."""
from os.path import join
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product
from statannotations.Annotator import Annotator
from scipy.stats import wilcoxon
from scripts import config as cfg
from scripts.cluster_stats import lineplot_compare
from scripts.corr_stats import (corr_freq_pvals, p_value_df, sample_size_df,
                                _get_freqs_correlation, _corr_results,
                                independent_corr,
                                _correct_sample_size)


def _save_fig(fig, fig_name, save_dir, close=True, transparent=False,
              facecolor=(1, 1, 1, 0), bbox_inches='tight'):
    """Save figure."""
    assert not fig_name.endswith(".png"), "You wanna save pdf not png!"
    if not fig_name.endswith(".pdf"):
        fig_name += ".pdf"
    save_path = join(save_dir, fig_name)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches=bbox_inches, transparent=transparent,
                facecolor=facecolor)
    if close:
        plt.close()


def plot_corrs(df, X, Y, hue=None, corr_method="spearman", figsize=None,
               color_labels=None, yticks=None, xticks=None, title=True,
               fig_name=None, save_dir=None, color_markers=None, leg_kws={},
               line_kws={}, scatter_kws={}, repeated_m='subject',
               fontsize=None, xlabels=None, ylabels=None, R2=False,
               n_perm=None, remove_ties=True, subs_special=[],
               add_sample_size=True,
               ci=None, scale=2, ylim=None, xlim=None, corr_comparison=False):
    label_dic = cfg.PLOT_LABELS

    if isinstance(X, str):
        X = [X]
    if isinstance(Y, str):
        Y = [Y]
    if not len(df.dropna(subset=X+Y)):
        print(df.head())
        raise ValueError(f"No finite data! x={X}, y={Y}, hue={hue}")
    n_cols = len(X)
    n_rows = len(Y)

    if figsize is None:
        figsize = (scale*n_cols, scale*n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, sharey='row', sharex='col',
                             figsize=figsize)
    axes = _axes2d(axes, n_rows, n_cols)
    for row_idx, y in enumerate(Y):
        for col_idx, x in enumerate(X):

            ax = axes[row_idx, col_idx]

            plot_corr(ax, df, x, y, corr_method=corr_method, hue=hue,
                      color_markers=color_markers, R2=R2,
                      corr_comparison=corr_comparison, n_perm=n_perm,
                      ci=ci, xlabel=None, leg_kws=leg_kws, line_kws=line_kws,
                      scatter_kws=scatter_kws, repeated_m=repeated_m,
                      remove_ties=remove_ties, subs_special=subs_special,
                      ylabel=None, title=None, leg_title=None,
                      add_sample_size=add_sample_size)

            try:
                label_color = color_labels[col_idx]
            except (IndexError, TypeError):
                label_color = "k"
            if yticks:
                ax.set_yticks(yticks)
            if xticks:
                ax.set_xticks(xticks)
            if col_idx == 0:
                ylabel = label_dic[y] if y in label_dic else y
                if ylabels is not None:
                    ylabel = ylabels[row_idx]
            else:
                ylabel = None
            if row_idx == len(Y) - 1:
                xlabel = label_dic[x] if x in label_dic else x
                if xlabels is not None:
                    xlabel = xlabels[col_idx]
            else:
                xlabel = None
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.set_ylabel(ylabel, c=label_color, fontsize=fontsize)
            ax.set_xlabel(xlabel, c=label_color, fontsize=fontsize)
            ax.yaxis.set_tick_params(which='both', labelbottom=True,
                                     labelsize=fontsize)
            ax.xaxis.set_tick_params(which='both', labelbottom=True,
                                     labelsize=fontsize)

    try:
        cond = df.cond.unique()
        psd_kind = df.psd_kind.unique()
        projects = df.project.unique()
        if len(projects) == 5:
            projects = ['all']
    except AttributeError:
        pass
    else:
        if hue == 'cond':
            add_cond = ''
            assert len(psd_kind) == 1, 'Dont mix psd kinds!'
            if len(projects) > 1:
                assert 'all' not in projects, 'Dont mix projects!'
            add_kind = psd_kind[0].capitalize()
            add_project = projects[0]
        elif hue == 'psd_kind':
            assert len(cond) == 1, 'Dont mix conditions!'
            add_cond = cond[0].upper()
            add_kind = ''
            add_project = ''
        elif hue == 'fm_params':
            assert len(cond) == 1, 'Dont mix conditions!'
            add_cond = cond[0].upper()
            add_kind = ''
            add_project = ''
        elif hue == 'project':
            assert len(cond) == 1, 'Dont mix conditions!'
            assert len(psd_kind) == 1, 'Dont mix psd kinds!'
            add_cond = cond[0].upper()
            add_kind = psd_kind[0].capitalize()
            add_project = ''
        elif hue is None:
            assert len(cond) == 1, 'Dont mix conditions!'
            assert len(psd_kind) == 1, 'Dont mix psd kinds!'
            if len(projects) > 1:
                assert 'all' not in projects, 'Dont mix projects!'
            add_cond = cond[0].upper()
            add_kind = psd_kind[0].capitalize()
            add_project = projects[0]

        if title is False:
            title = ""
        elif title is True:
            title = f'{add_cond} {add_kind} {add_project}'
    plt.suptitle(title)
    plt.tight_layout()
    if fig_name:
        if save_dir is None:
            save_dir = cfg.FIG_RESULTS
        _save_fig(fig, fig_name, save_dir, close=False)
    else:
        plt.show()
    return fig, axes


def explode_df(df, freqs="psd_freqs", psd="asd", fm_params="broad",
               drop_cols=True, keep_cols=[], fmax=cfg.LOWPASS, fmin=0):
    if isinstance(keep_cols, str):
        keep_cols = [keep_cols]
    if isinstance(fm_params, str):
        fm_params = [fm_params]
    # remove duplicates:
    fm_params_all = df.fm_params.unique()
    if ("fm_params" in df.columns) and (len(fm_params_all) > 1):
        if fm_params in fm_params_all:
            df = df[df.fm_params.isin(fm_params)]
        else:
            df = df[df.fm_params.isin(fm_params_all[0])]
    # Ignore LFP 4-5, 5-6
    ignore_chs = ["LFP_4-5", "LFP_5-6", "LFP_5", "LFP_6"]
    df = df[~df.ch.isin(ignore_chs)]

    if drop_cols:
        keep = ["subject", "cond", "ch", "title", "bids_basename",
                "ch_directional", "ch_hemisphere", "ch_nme",
                "ch_reference", 'psd_kind',
                "ch_type_bids", "sweet_spot_distance",
                "ch_sweetspot",
                "project"] + [psd, freqs] + keep_cols
        df = df[keep]
        # Create df where each PSD value is in a separate row
    df_psd = df[df[psd].notna()].explode([freqs, psd])
    df_psd[freqs] = df_psd[freqs].astype(int)
    # Only consider freqs up to 500 Hz
    df_psd = df_psd[(df_psd[freqs] >= fmin) & (df_psd[freqs] <= fmax)]
    df_psd = df_psd.reset_index(drop=True)
    return df_psd


def plot_clouds(df, Y, x='cond', order=['off', 'ON'], col=None,
                yscale="linear", same_subjects=True, fig_name=None,
                palette=None, combine_projects=True, col_title=False,
                dispersion=False):
    """Plot rainclouds subplots. Don't enable hue, use separate subplot
    for each project."""
    if any('abs' in y for y in Y):
        assert len(df.psd_kind.unique()) == 1, "Dont mix psd kinds!"
    kind = df.psd_kind.unique()[0]
    if x == "cond":
        # rename cond column to LDOPA
        df = df.set_index(x).rename(index=cfg.COND_DICT).reset_index()
        df = df.rename(columns={"cond": "LDOPA"})
        x = "LDOPA"
        order = ["off", "on"]
    df = df[df[x].isin(order)]

    assert len(df), "No data!"
    test = "Wilcoxon" if same_subjects else 'Mann-Whitney'

    if isinstance(Y, str):
        Y = [Y]
    n_rows = len(Y)
    if col:
        cols = df[col].unique()
        if col == "project":
            cols = [col for col in cfg.PROJECT_ORDER if col in cols]
        col_slices = [df[col] == col_val for col_val in cols]
        if combine_projects and col == "project":
            cols.append('all')
            col_slices.append(slice(None))
    else:
        cols = [None]
        col_slices = [slice(None)]
    n_cols = len(cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 3.5*n_rows),
                             sharex="col", sharey="row")
    axes = _axes2d(axes, n_rows, n_cols)
    for row_idx, y in enumerate(Y):
        ymins = []
        ymaxs = []
        for col_idx, col_slice in enumerate(col_slices):
            df_plot = df[col_slice]
            ax = axes[row_idx, col_idx]
            if col_idx == 0:
                ylabel = y.replace("_abs_max", "").replace('_log', '')
                ylabel = ylabel.replace("_freq", "").replace("_fm", "")
                ylabel = ylabel.replace("_auc", "").replace("_nonoise", "")
                ylabel = ylabel.replace("_max", "").replace("powers", "")
                ylabel = ylabel.replace("centerfreqs", "")
                ylabel = ylabel.strip("_")
                try:
                    ylabel = cfg.BAND_NAMES[ylabel]
                except KeyError:
                    try:
                        ylabel = cfg.PLOT_LABELS_UNITS[ylabel]
                    except KeyError:
                        ylabel = y
                if 'freq' in y:
                    ylabel += " Frequency [Hz]"
            else:
                ylabel = None
            col = cols[col_idx]
            if row_idx == 0:
                try:
                    title = cfg.PROJECT_DICT[col]
                except KeyError:
                    title = col if col_title else None
            else:
                title = None
            xlabel = x if row_idx == n_rows - 1 else None
            try:
                palette = (cfg.COLOR_DIC[col], cfg.COLOR_DIC[col + "2"])
            except KeyError:
                palette = None
            ax.set_yscale(yscale)
            _add_raincloud_and_stats(df_plot, ax, x, y, test=test, order=order,
                                     palette=palette, ylabel=ylabel,
                                     xlabel=xlabel, title=title,
                                     dispersion=dispersion)
            # why does automatic ylim not work?
            ymins.append(df_plot[y].min())
            _, ymax = ax.get_ylim()
            ymaxs.append(ymax)
        ymin = min(ymins)
        ymax = max(ymaxs)
        ymin -= np.abs(ymin * .02)
        ymax += np.abs(ymax * .03)
        ymin = -.1 if ymin == 0 else ymin
        ymin = -5 if y == 'fm_knee_fit' else ymin
        ymin = None if np.isnan(ymin) else ymin
        [ax.set_ylim(ymin, ymax) for ax in axes[row_idx]]
    test_nme = "Wilcoxon signed-rank test" if same_subjects else test
    units = _units_from_y_kind(y, kind)
    kind = 'Normalized' if kind == 'normalized' else 'Absolute'
    title = f"{test_nme}, {kind} {units}"
    plt.suptitle(title)
    plt.tight_layout()
    if fig_name:
        _save_fig(fig, fig_name, cfg.FIG_RESULTS)
    else:
        plt.show()


def _axes2d(axes, n_rows, n_cols):
    if n_rows == n_cols == 1:
        axes = np.array([[axes]])  # convert ax to 2d array
    elif n_rows == n_cols == 2:
        assert axes.ndim == 2
    elif n_rows == 1:
        axes = axes.reshape(1, -1)  # convert 1d to 2d array
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)  # convert 1d to 2d array
    return axes


def _units_from_y_kind(y, kind):
    pwr_lin_units = r'[$\mu V^2/Hz$]'
    pwr_log_units = r'[log10$(\mu V^2/Hz)$]'
    freq_units = '[$Hz$]'
    log_powers = ['fm_offset_log', '_abs_max_log', '_abs_max_log_nonoise']
    lin_powers = ['_fm_powers_max', 'fm_offset', '_abs_max',
                  '_abs_max_nonoise']
    if 'freq' in y or y == 'fm_knee_fit':
        return freq_units
    if kind == 'normalized' and 'log' in y:
        return '[log10(%)]'
    elif kind == 'normalized' and 'log' not in y:
        return '[%]'
    if y.endswith('fm_auc_log'):
        return r'[log10$(\mu V^2) - 1$]'
    elif y.endswith('fm_auc'):
        return r'[$\mu V^2$]'
    elif any(y.endswith(log_pwr) for log_pwr in log_powers):
        return pwr_log_units
    elif any(y.endswith(lin_pwr) for lin_pwr in lin_powers):
        return pwr_lin_units
    elif y.endswith('_fm_powers_log_max'):
        return r'[log10$(\mu V^2/Hz) - 1$]'
    elif y.startswith('fm_exponent'):
        return ''
    elif y.startswith('mni'):
        return r'[$mm$]'
    elif y == 'fm_knee' or y.endswith('fm_peak_count') or 'ratio' in y:
        return ''
    else:
        raise ValueError(f"Unknown kind of y: {y}")


def stacked_bar_chart(df, bands, fig_name=None):
    """Plot ratios of frequency bands standard vs normalized as pie chart or
    stacked bar chart.

    Questions:
    Only alpha, low-beta, high-beta or others too?
    Make bar chart same height? Or different heights to show band changes
    outside of normalization range?"""
    sns.set()
    df = df[df.cond.isin(['on', 'off'])]

    Y_lin = [f"{band}_abs_max" for band in bands]
    data = df.groupby(['psd_kind', 'cond'])[Y_lin].mean().reset_index()

    # Separating data for 'standard' and 'normalized' conditions,
    # for both 'off' and 'on' conditions
    stand = data['psd_kind'] == 'standard'
    norm = data['psd_kind'] == 'normalized'
    on = data['cond'] == 'on'
    off = data['cond'] == 'off'
    drop = ['psd_kind', 'cond']
    data_standard_off = data[stand & off].drop(columns=drop).squeeze(axis=0)
    data_normalized_off = data[norm & off].drop(columns=drop).squeeze(axis=0)
    data_standard_on = data[stand & on].drop(columns=drop).squeeze(axis=0)
    data_normalized_on = data[norm & on].drop(columns=drop).squeeze(axis=0)

    # Normalizing data for pie charts
    data_standard_off = data_standard_off / data_standard_off.sum() * 0.5
    data_normalized_off = data_normalized_off / data_normalized_off.sum() * 0.5
    data_standard_on = data_standard_on / data_standard_on.sum() * 0.5
    data_normalized_on = data_normalized_on / data_normalized_on.sum() * 0.5

    # Concatenating data for outer and inner pie charts
    outer_pie_data = pd.concat([data_standard_off, data_normalized_off[::-1]])
    inner_pie_data = pd.concat([data_standard_on, data_normalized_on[::-1]])

    # Labels and colors
    # this needs to be automized to avoid errors of labels and colors
    colors_off = ['#377eb8', '#4daf4a', '#e41a1c',
                  '#e41a1c', '#4daf4a', '#377eb8']
    colors_on = ['#8da0cb', '#b3de69', '#fb8072',
                 '#fb8072', '#b3de69', '#8da0cb']

    # Creating the nested pie chart
    fig = plt.figure(figsize=(8, 8))
    pie_kwargs = dict(startangle=90, counterclock=False,
                      wedgeprops=dict(width=0.3, edgecolor='w'),
                      autopct='%1.1f%%', pctdistance=0.85,
                      textprops={'color': 'white'})
    plt.pie(outer_pie_data, colors=colors_off, radius=1.0, **pie_kwargs)
    plt.pie(inner_pie_data, colors=colors_on, radius=0.7, **pie_kwargs)

    # Adding a legend
    legend_labels = ['Alpha', 'Low Beta', 'High Beta']
    plt.legend(legend_labels, loc='upper right')

    # Adding annotations
    plt.annotate('Standard', xy=(1.2, 1.1), xytext=(0.1, 1.1), ha='left')
    plt.annotate('Normalized', xy=(-1.2, 1.1), xytext=(-0.1, 1.1), ha='right')
    plt.annotate('off', xy=(0.5, 0), xytext=(1.1, 0), ha='center', va='center')
    plt.annotate('ON', xy=(0, 0), xytext=(0.3, 0), ha='center', va='center')
    plt.tight_layout()
    if fig_name:
        _save_fig(fig, fig_name, cfg.FIG_RESULTS)
    else:
        plt.show()


def plot_clouds_single_ax(df, Y, hue='cond', hue_order=['off', 'ON'],
                          x='project', same_subjects=True, palette=None):
    """Plot rainclouds subplots with x=project and hue=cond. Unfortunaley,
    does not solve different offsets for statannotations, therefore no
    advantages."""
    assert len(df.psd_kind.unique()) == 1, "Dont mix psd kinds!"
    kind = df.psd_kind.unique()[0]
    if isinstance(Y, str):
        Y = [Y]
    if hue == "cond":
        # rename cond column to LDOPA
        df = df.set_index(hue).rename(index={"on": 'on', "off": 'off'})
        df = df.reset_index().rename(columns={"cond": "LDOPA"})
        hue = "LDOPA"
        hue_order = ['off', 'on']
    df = df[df[hue].isin(hue_order)]

    assert len(df), "No data!"
    test = "Wilcoxon" if same_subjects else 'Mann-Whitney'

    n_rows = len(Y)
    if x == "project":
        order = df[x].unique()
        order = [col for col in cfg.PROJECT_ORDER if col in order]
        palette = [(cfg.COLOR_DIC[proj], cfg.COLOR_DIC[proj + "2"])
                   for proj in order]
    n_cols = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows),
                             sharex="col", sharey="row")
    axes = _axes2d(axes, n_rows, n_cols)
    for row_idx, y in enumerate(Y):
        ax = axes[row_idx]
        ylabel = y.replace("_abs_max_log", "")
        ylabel = cfg.BAND_NAMES[ylabel]
        xlabel = hue if row_idx == n_rows - 1 else None
        _add_raincloud_and_stats_single(df, ax, x, y, test=test, order=order,
                                        hue=hue, hue_order=hue_order,
                                        palette=palette, ylabel=ylabel,
                                        xlabel=xlabel)
    test_nme = "Wilcoxon signed-rank test" if same_subjects else test
    kind = 'Not normalized' if kind == 'standard' else 'Normalized'
    ylabel = r'Max. PSD [log10$(\mu V^2/Hz)$]'
    title = f"{test_nme}, {kind}, {ylabel}"
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def _add_raincloud_and_stats_single(df, ax, x, y, test="Wilcoxon", order=None,
                                    hue=None, hue_order=None, palette=None,
                                    xlabel=None, ylabel=None, title=None):
    import ptitprince as pt
    from statannotations.Annotator import Annotator
    # give effect sizes without equal sample sizes
    effect_sizes = []
    for project in order:
        group = df[df.project == project][[hue, y]].groupby(hue)
        x1, x2 = group[y].apply(lambda x: x.values)
        effect_sizes.append(cohen_d(x1, x2))

    if test == "Wilcoxon":
        df, n = equalize_x_and_y(df, hue, y)
    else:
        n = len(df[y].dropna()) // len(df[x].dropna().unique())
    plotting_parameters = {'ax': ax, 'data': df, 'x': x, 'y': y, "hue": hue,
                           'order': order, "hue_order": hue_order,
                           'palette': palette}
    pt.RainCloud(width_viol=.3, width_box=.25, **plotting_parameters,
                 dodge=True, alpha=.5)
    pairs = [[(proj, 'off'), (proj, 'on')] for proj in order]
    annotator = Annotator(pairs=pairs, **plotting_parameters)
    annotator.configure(test=test, text_format='simple', loc='inside',
                        verbose=False, show_test_name=False)
    annotator.apply_test()
    _custom_annotations(annotator, effect_sizes, n)

    ax.set_title(title)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    return ax


def _add_raincloud_and_stats(df, ax, x, y, test="Wilcoxon", order=None,
                             palette=None, xlabel=None, ylabel=None,
                             title=None, dispersion=False):
    import ptitprince as pt
    from statannotations.Annotator import Annotator

    plot_params = {'ax': ax, 'data': df, 'x': x, 'y': y, 'order': order,
                   'palette': palette, 'pointplot': False,
                   'box_showfliers': False}
    pt.RainCloud(width_viol=.3, width_box=.25, **plot_params)
    effect_size, n_cohen = _get_cohen_stats(df, x, y)
    if dispersion:
        arr_off = df[df[x] == order[0]][y].dropna()
        dispersion = quartile_coefficient_of_dispersion(arr_off)
    else:
        dispersion = None

    stat_params = plot_params
    if test == "Wilcoxon":
        df, n_pvalue = equalize_x_and_y(df, x, y)
        stat_params['data'] = df
        both_conds = np.isfinite(n_pvalue)
    else:
        both_conds = np.isfinite(effect_size)
        n_pvalue = n_cohen
    if both_conds:
        annotator = Annotator(pairs=[order], **stat_params)
        annotator.configure(test=test, text_format='simple', loc='inside',
                            verbose=False, show_test_name=False)
        try:
            annotator.apply_test()
        except ValueError:
            pass
        else:
            _custom_annotations(annotator, effect_size, n_cohen, n_pvalue,
                                dispersion)

    ax.set_title(title)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    return ax


def _custom_annotations(annotator, effect_sizes, n_cohen, n_pvalue,
                        dispersions=None):
    if isinstance(effect_sizes, float):
        effect_sizes = [effect_sizes]
    if isinstance(dispersions, float):
        dispersions = [dispersions]
    msg = "Need effect size for each pair!"
    assert len(effect_sizes) == len(annotator.pairs), msg
    annos_new = []
    for idx, anno in enumerate(annotator.get_annotations_text()):

        # add p-value anno
        wilcoxon_str = anno.replace("p = ", "p=")  # kill spaces
        wilcoxon_str = wilcoxon_str.replace("p ≤ ", "p≤")
        if annotator.annotations[idx].data.is_significant:
            wilcoxon_str = rf'$\bf{{{wilcoxon_str}}}$'  # make bold

        # add cohen's d anno
        effect_size = effect_sizes[idx]
        if np.abs(effect_size) >= 0.5:
            effect_size_str = fr"$\bf{{d={effect_size:.2f}}}$"  # make bold
        else:
            effect_size_str = f"d={effect_size:.2f}"

        # add dispersion anno
        if dispersions:
            dispersion = f'\nQCD={dispersions[idx]:.2f}'
        else:
            dispersion = ""

        if isinstance(n_pvalue, tuple):
            sample_str = f"(n1={n_pvalue[0]}, n2={n_pvalue[1]})"
            full_anno = f'{wilcoxon_str}, {effect_size_str}\n{sample_str}'
        elif isinstance(n_pvalue, int):
            sample_str_wil = f"(n={n_pvalue})"
            sample_str_coh = f"(n={np.mean(n_cohen)})"
            full_anno = (f'{wilcoxon_str} {sample_str_wil}\n'
                         f'{effect_size_str} {sample_str_coh}')
        full_anno += dispersion
        annos_new.append(full_anno)
    annotator.annotate_custom_annotations(annos_new)


def _get_cohen_stats(df, x, y):
    df = df[[x, y]].dropna()
    try:
        x1, x2 = df.groupby(x)[y].apply(lambda x: x.values)
    except ValueError:
        # only one condition
        effect_size, n_cohen = np.nan, np.nan
    else:
        effect_size = cohen_d(x1, x2)
        n_cohen = (len(x1), len(x2))
    return effect_size, n_cohen


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    x_std_sq = np.std(x, ddof=1)**2
    y_std_sq = np.std(y, ddof=1)**2
    mean_difference = np.mean(x) - np.mean(y)
    std = np.sqrt(((nx-1) * x_std_sq + (ny-1) * y_std_sq) / dof)
    return mean_difference / std


def quartile_coefficient_of_dispersion(array):
    if not len(array):
        return np.nan
    q1, q3 = np.percentile(array, [25, 75])
    return (q3 - q1) / (q3 + q1)


def equalize_x_and_y(df, x, y) -> tuple[pd.DataFrame, int]:
    """Wilcoxon requires equal sample sizes."""
    if x:
        df = df.dropna(subset=[x, y])
    elif y:
        df = df.dropna(subset=y)
        return df, len(df)

    if x == 'cond':
        df = df[df.cond.isin(['off', 'on', 'off', 'ON'])]
    # ch_hemisphere seems more reasonable in case of channel switching on ON
    # and OFF condition. Can be problematic though when all contacts are
    # evaluated.
    # group = ["subject", 'ch_hemisphere']
    group = ["subject", 'ch_nme']
    hemi_both_conds = df.groupby(group)[x].nunique() == df[x].nunique()
    hemi_both_conds = hemi_both_conds.reindex(df.set_index(group).index)
    df = df.set_index(group)[hemi_both_conds].reset_index()

    try:
        sample_size = len(df[y]) // len(df[x].unique())
    except ZeroDivisionError:
        sample_size = np.nan
    return df, sample_size


def _plot_ax_psd_freqs(df, ax, x_plot, y_plot, y_pval=None,
                       title=None, color=None):

    sns.lineplot(data=df, x=x_plot, y=y_plot, ax=ax, color=color,
                 label=r"$\rho$")
    if y_pval:
        sns.lineplot(data=df, x=x_plot, y=y_pval, ax=ax, color="grey",
                     linestyle="--", ci=None, label="p-value")
    ax.hlines(0.05, *ax.get_xlim(), color="grey", linestyle="--", lw=.5,
              label="0.05")
    corr_freq_pvals(ax, df, x_plot, y_pval, y_height=0)

    ax.legend()
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(y_plot)
    ax.set_title(title)


def _plot_band(band, ylim, ax):
    if band:
        band = cfg.BANDS[band]
        band_start = band[0]
        band_len = np.diff(band)
        if not ylim:
            ylim = ax.get_ylim()
        rect = plt.Rectangle((band_start, ylim[0]), band_len,
                             np.diff(ylim), color="r", alpha=0.1)
        ax.add_patch(rect)
    return ylim


def plot_psd_df_cluster(df, hue, x='psd_freqs', y='psd',
                        x_min=1, x_max=45, alpha=0.05, n_perm=100000,
                        xscale="linear", yscale="log", y_label=None,
                        title=None, save_name=None,
                        anno_buffer=3.5,  # depends (xmax, fontsize, figsize)
                        x_label="Frequency [Hz]", print_clust_border=True
                        ) -> None:
    if hue.endswith("_thirds"):
        df = df[df[hue] != 'moderate_third']
    elif hue.endswith("_quartiles"):
        df = df[~df[hue].isin(['Q2', 'Q3'])]
    conds = df[hue].dropna().unique()
    assert len(conds) == 2, f"Need 2 conditions! Got {conds}"
    df1 = df[df[hue] == conds[0]]
    df2 = df[df[hue] == conds[1]]

    times = df[x].values[0]
    x_mask = (times >= x_min) & (times <= x_max)
    times = times[x_mask]

    x1 = np.stack(df1[y].values)[:, x_mask].T
    x2 = np.stack(df2[y].values)[:, x_mask].T

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), squeeze=True)
    if not y_label:
        y_label = y
    _, cluster_borders = lineplot_compare(ax=ax, x_1=x1, x_2=x2,
                times=times,
                data_labels=conds,
                x_label=x_label, y_label=y_label, alpha=alpha,
                n_perm=n_perm, correction_method="cluster_pvals",
                two_tailed=True, paired_x1x2=False,
                title=title, show=False)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    if print_clust_border and cluster_borders:
        height = ax.get_ylim()[0]
        lim1 = cluster_borders[0][0]
        lim2 = cluster_borders[-1][-1]
        text1 = f"{lim1:.0f} Hz"
        text2 = f"{lim2:.0f} Hz"
        diff = lim2 - lim1
        kwargs = dict(horizontalalignment="left",
                      verticalalignment="bottom")
        if diff < 12:
            text1 = text1.replace(' Hz', '')
            lim1 -= 1/diff * anno_buffer
            lim2 += 1/diff * anno_buffer
            ax.text((lim1 + lim2) / 2, height, ' - ', **kwargs)
        ax.text(lim1, height, text1, **kwargs)
        ax.text(lim2, height, text2, **kwargs)
    plt.tight_layout()
    if save_name:
        _save_fig(fig, save_name, cfg.FIG_RESULTS)
    else:
        plt.show()


def plot_corr(ax, df, x, y, corr_method='within', hue="cond", ci=None,
              repeated_m="subject", color_markers=None, R2=False,
              xlabel=None, corr_comparison=False, marker='o',
              ylabel=None, title=None, leg_title=None, n_perm=None,
              scatter_kws=None, remove_ties=True, subs_special=[],
              line_kws={}, leg_kws={}, add_sample_size=True):
    # repeated_m gets ignored if corr_method is not rmcorr
    if y == 'UPDRS_III':
        if (corr_method == 'within') and (repeated_m == 'subject'):
            msg = "repeated_m='subject' does not work for UPDRS_III!"
            raise ValueError(msg)
        # average hemispheres
        keep = ['subject', 'cond', 'project', 'color']
        df = df.groupby(keep).mean(numeric_only=True).reset_index()

    df = df.dropna(subset=[x, y])

    if hue:
        df_rho = df.set_index(hue)
        if hue == 'cond':
            hue_order = cfg.COND_ORDER
        elif hue == 'project':
            hue_order = cfg.PROJECT_ORDER
        else:
            hue_order = df_rho.index.unique()
    else:
        df_rho = df
        hue_order = [slice(None)]

    rhos = []
    sample_sizes = []
    labels = []
    handles = []
    weights = []
    for row_idx in hue_order:
        if hue is None:
            pass
        elif row_idx not in df_rho.index:
            continue
        if corr_method == 'withinRank':
            df_rho = _correct_sample_size(df_rho, x, y, repeated_m=repeated_m,
                                          remove_ties=remove_ties)
        corr_results = _corr_results(df_rho, x, y, corr_method, row_idx,
                                     n_perm=n_perm, remove_ties=remove_ties,
                                     add_sample_size=add_sample_size,
                                     repeated_m=repeated_m, R2=R2)
        rho, sample_size, label, weight, _ = corr_results

        rhos.append(rho)
        sample_sizes.append(sample_size)
        labels.append(label)
        weights.append(weight)
        if color_markers is None:
            try:
                color_markers_plot = cfg.COLOR_DIC[row_idx]
            except (KeyError, TypeError):
                color_markers_plot = None
        else:
            color_markers_plot = color_markers

        if corr_method.startswith('within'):
            if scatter_kws is None:
                scatter_kws = {}
            plot_rm_corr(ax=ax, data=df_rho.loc[row_idx], x=x, y=y,
                         subject=repeated_m, subs_special=subs_special,
                         scatter_kws=scatter_kws, line_kws=line_kws
                         )
        else:
            if len(subs_special):
                raise NotImplementedError("subs_special not implemented!")
            df_plot = df_rho.loc[row_idx].dropna(subset=[x, y])
            if scatter_kws is None:
                scatter_kws_plot = dict(color=list(df_plot.color.values))
            else:
                scatter_kws_plot = scatter_kws
            sns.regplot(ax=ax, data=df_plot, y=y, x=x, ci=ci,
                        color=color_markers_plot, label=label, marker=marker,
                        scatter_kws=scatter_kws_plot, line_kws=line_kws)
        handle, _ = ax.get_legend_handles_labels()
        handles.append(handle[-1])

    if hue is None:
        assert title is None, "You have a problem"
        assert len(labels) == len(weights) == 1, "You have a problem"
        ax.set_title(labels[0], weight=weights[0])
    else:
        _plot_legend(ax, x, y, labels, weights, hue, rhos, sample_sizes,
                     handles, leg_title,
                     xylabels=False, leg_kws=leg_kws,
                     corr_comparison=corr_comparison)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax


def _plot_legend(ax, X, Y, labels, weights, hue, rhos, sample_sizes,
                 handles=None,
                 leg_title=None, xylabels=True, leg_kws={},
                 corr_comparison=False, title_long=True,
                 borderaxespad=0, handletextpad=0, borderpad=0.2,
                 mode='expand', bbox_to_anchor=(0, 1.02, 1, .2),
                 loc='lower left', ncol=1):

    title, title_fontproperties = _leg_titles(rhos, sample_sizes, leg_title,
                                              corr_comparison=corr_comparison,
                                              title_long=title_long)
    if title is not None:
        leg_kws['title'] = title
    handlelength = None if hue else 2

    # overwrite all settings that are present in leg_kws
    for key, value in leg_kws.copy().items():
        if key == 'title':
            title = value
            leg_kws.pop(key)
            continue
        if key == 'title_fontproperties':
            title_fontproperties = value
            leg_kws.pop(key)
            continue
        if key == 'handlelength':
            handlelength = value
            leg_kws.pop(key)
            continue
        if key == 'borderaxespad':
            borderaxespad = value
            leg_kws.pop(key)
            continue
        if key == 'handletextpad':
            handletextpad = value
            leg_kws.pop(key)
            continue
        if key == 'borderpad':
            borderpad = value
            leg_kws.pop(key)
            continue
        if key == 'mode':
            mode = value
            leg_kws.pop(key)
            continue
        if key == 'bbox_to_anchor':
            bbox_to_anchor = value
            leg_kws.pop(key)
            continue
        if key == 'loc':
            loc = value
            leg_kws.pop(key)
            continue
        if key == 'ncol':
            ncol = value
            leg_kws.pop(key)

    leg = ax.legend(loc=loc, ncol=ncol, handles=handles,
                    labels=labels, handlelength=handlelength,
                    borderaxespad=borderaxespad,
                    handletextpad=handletextpad,
                    borderpad=borderpad,
                    mode=mode,
                    title_fontproperties=title_fontproperties,
                    bbox_to_anchor=bbox_to_anchor, **leg_kws)

    # Set significant legend elements bold
    [t.set_fontweight(w) for t, w in zip(leg.get_texts(), weights)]
    if xylabels:
        ax.set_xlabel(X.replace("_", " ").capitalize())
        ax.set_ylabel(Y.replace("_", " ").capitalize())


def plot_rm_corr(
    data=None,
    ax=None,
    x=None,
    y=None,
    subject=None,
    line_kws={},
    scatter_kws={},
    subs_special=[],
    lw=.2
):
    """Plot a repeated measures correlation.

    >>>EDITED FROM pg.plot_rm_corr<<<


    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Dataframe.
    x, y : string
        Name of columns in ``data`` containing the two dependent variables.
    subject : string
        Name of column in ``data`` containing the subject indicator.
    legend : boolean
        If True, add legend to plot. Legend will show all the unique values in
        ``subject``.
    kwargs_facetgrid : dict
        Optional keyword arguments passed to :py:class:`seaborn.FacetGrid`
    kwargs_line : dict
        Optional keyword arguments passed to :py:class:`matplotlib.pyplot.plot`
    kwargs_scatter : dict
        Optional keyword arguments passed to
        :py:class:`matplotlib.pyplot.scatter`

    Returns
    -------
    g : :py:class:`seaborn.FacetGrid`
        Seaborn FacetGrid.

    See also
    --------
    rm_corr

    Notes
    -----
    Repeated measures correlation [1]_ (rmcorr) is a statistical technique
    for determining the common within-individual association for paired
    measures assessed on two or more occasions for multiple individuals.

    Results have been tested against the
    `rmcorr <https://github.com/cran/rmcorr>` R package. Note that this
    function requires `statsmodels
    <https://www.statsmodels.org/stable/index.html>`_.

    Missing values are automatically removed from the ``data``
    (listwise deletion).

    References
    ----------
    .. [1] Bakdash, J.Z., Marusich, L.R., 2017. Repeated Measures Correlation.
           Front. Psychol. 8, 456. https://doi.org/10.3389/fpsyg.2017.00456

    Examples
    --------
    Default repeated measures correlation plot

    .. plot::

        >>> import pingouin as pg
        >>> df = pg.read_dataset('rm_corr')
        >>> g = pg.plot_rm_corr(data=df, x='pH', y='PacO2', subject='Subject')

    With some tweakings

    .. plot::

        >>> import pingouin as pg
        >>> import seaborn as sns
        >>> df = pg.read_dataset('rm_corr')
        >>> sns.set(style='darkgrid', font_scale=1.2)
        >>> g = pg.plot_rm_corr(data=df, x='pH', y='PacO2',
        ...                     subject='Subject', legend=True,
        ...                     kwargs_facetgrid=dict(height=4.5, aspect=1.5,
        ...                                           palette='Spectral'))
    """
    # Check that stasmodels is installed
    from pingouin.utils import _is_statsmodels_installed

    _is_statsmodels_installed(raise_error=True)
    from statsmodels.formula.api import ols

    # Safety check (duplicated from pingouin.rm_corr)
    assert isinstance(data, pd.DataFrame), "Data must be a DataFrame"
    assert x in data.columns, "The %s column is not in data." % x
    assert y in data.columns, "The %s column is not in data." % y
    assert data[x].dtype.kind in "bfiu", "%s must be numeric." % x
    assert data[y].dtype.kind in "bfiu", "%s must be numeric." % y
    assert subject in data.columns, "The %s column is not in data." % subject
    if data[subject].nunique() < 3:
        raise ValueError("rm_corr requires at least 3 unique subjects.")

    # Correct sample size - at least 2 datapoints per subject
    data = _correct_sample_size(data, x, y, repeated_m=subject)

    # Fit ANCOVA model
    # https://patsy.readthedocs.io/en/latest/builtins-reference.html
    # C marks the data as categorical
    # Q allows to quote variable that do not meet Python variable name rule
    # e.g. if variable is "weight.in.kg" or "2A"
    assert x not in ["C", "Q"], "`x` must not be 'C' or 'Q'."
    assert y not in ["C", "Q"], "`y` must not be 'C' or 'Q'."
    assert subject not in ["C", "Q"], "`subject` must not be 'C' or 'Q'."
    formula = f"Q('{y}') ~ C(Q('{subject}')) + Q('{x}')"
    model = ols(formula, data=data).fit()

    # Fitted values
    data = data.copy()
    data["pred"] = model.fittedvalues

    data['line_width'] = lw
    subjects = data[subject].unique()
    n_subs = len(subjects)
    markers = {sub: "." for sub in subjects}
    if subject == 'project':
        markers = True  # use default markers for projects

    # Emphasize special subjects
    data['subs_special'] = data[subject].isin(subs_special)

    # Use larger linewidths and points for special subjects
    # data.loc[~data.subs_special, 'line_width'] = lw
    data.loc[data.subs_special, 'line_width'] = 4 * lw
    point_size_normal = 10
    point_size_special = 8
    # Define order of plotting:
    # normal scatter → normal lines → special lines → special scatter
    zorder_scatter_normal = 1
    data.loc[~data.subs_special, 'zorder_lines'] = 2
    data.loc[data.subs_special, 'zorder_lines'] = 3
    zorder_scatter_special = 4
    if len(subs_special):

        # Use different markers for special subjects
        subjects = data[subject].unique()
        marker_styles = cfg.SYMBOLS_SPECIAL_SUBS
        assert len(marker_styles) >= len(subs_special), "Too many special subs!"
        markers_special = {sub: marker_styles[i] for i, sub
                           in enumerate(subs_special)}
        markers.update(markers_special)

        # Change order of subjects to plot special subs last
        normal_subs = data[~data.subs_special].subject.unique().tolist()
        # indicated_subs = subs_special
        hue_order = normal_subs + subs_special
        data = pd.concat([data[~data.subs_special], data[data.subs_special]])

        # Use stronger colors for special subjects
        c_normal = sns.color_palette("pastel", len(normal_subs))
        palette = dict(zip(normal_subs, c_normal))
        palette.update(dict(zip(subs_special, cfg.COLORS_SPECIAL_SUBS)))
    else:
        palette = dict(zip(subjects, sns.color_palette("muted", n_subs)))
        hue_order = subjects

    # Plot indicated subject regression lines
    for sub in hue_order:
        data_sub = data[data[subject] == sub]
        line_width = data_sub.line_width.values[0]
        line_kws['lw'] = line_width
        line_kws['zorder'] = data_sub.zorder_lines.values[0]
        sns.regplot(x=x, y="pred", data=data_sub, ax=ax,
                    scatter=False,
                    ci=None, truncate=True,
                    color=palette[sub],
                    line_kws=line_kws,
                    scatter_kws=scatter_kws,
                    )
    # Plot scatter separately to avoid straight lines between scatter points
    scatter_kws['zorder'] = zorder_scatter_normal
    sns.scatterplot(x=x, y=y, data=data[~data.subs_special], ax=ax,
                    hue=subject, palette=palette, s=point_size_normal,
                    style='subject', markers=markers,
                    **scatter_kws
                    )
    scatter_kws['zorder'] = zorder_scatter_special
    if not data[data.subs_special].empty:
        sns.scatterplot(x=x, y=y, data=data[data.subs_special], ax=ax,
                        hue=subject, palette=palette, s=point_size_special,
                        style='subject', markers=markers,
                        **scatter_kws
                        )
    ax.legend().remove()
    return ax


def plot_psd_corr_freqs(df, x="psd", col='cond',
                        row='UPDRS_bradykinesia_contra', x_max=95,
                        figsize=None, save_name=None, corr_method='spearman',
                        rolling_mean=None):
    if isinstance(row, str):
        Y = [row]
    else:
        Y = row
    keep_cols = [x, col] + Y + ['psd_kind', 'fm_params', 'project',
                                'ch_hemisphere', "psd_freqs", "subject", 'ch']
    finite_cols = [col, x] + Y
    X = df[finite_cols].dropna().cond.unique()
    n_cols = len(X)
    n_rows = len(Y)
    if n_cols == 0 or n_rows == 0:
        return None

    df = df[keep_cols]

    if figsize is None:
        figsize = (4*n_cols, 4*n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey='row',
                             sharex=False)
    axes = _axes2d(axes, n_rows, n_cols)
    for row_idx, y_row in enumerate(Y):
        for col_idx, x_col in enumerate(X):

            ax = axes[row_idx, col_idx]
            df_col = df[df.cond == x_col]

            if y_row == 'UPDRS_III':
                hemi_mean = True
            else:
                hemi_mean = False
            df_corr = _get_freqs_correlation(df_col, x, y_row, x_max=x_max,
                                             corr_method=corr_method,
                                             average_hemispheres=hemi_mean)
            assert len(df.psd_kind.unique()) == 1
            kind = df.psd_kind.unique()[0]
            project = df.project.unique()[0]
            x_plot = f"{x}_freqs"
            y_plot = f"corr_{x}_{y_row}"
            y_pval = f"pval_{x}_{y_row}"

            if rolling_mean:
                rol_mean = df_corr[y_plot].rolling(rolling_mean,
                                                   center=True).mean()
                pval_mean = df_corr[y_pval].rolling(rolling_mean,
                                                    center=True).mean()
                y_plot = f"{y_plot}_rolling{rolling_mean}"
                y_pval = f"{y_pval}_rolling{rolling_mean}"
                df_corr[y_plot] = rol_mean
                df_corr[y_pval] = pval_mean

            title = x_col if row_idx == 0 else None
            color = cfg.COLOR_DIC[x_col]
            _plot_ax_psd_freqs(df_corr, ax, x_plot, y_plot, y_pval,
                               color=color, title=title)
    suptitle = f'{kind} {cfg.PROJECT_DICT[project]}'
    if rolling_mean:
        suptitle += rf" f\Delta f={rolling_mean} Hz$"
    plt.suptitle(suptitle)
    plt.tight_layout()
    if save_name:
        save_name += f"_x_max={df_corr.psd_freqs.max()}Hz"
        if rolling_mean:
            save_name += f"_rolling{rolling_mean}Hz"
        _save_fig(fig, save_name, cfg.FIG_RESULTS)
    else:
        plt.show()


def plot_psd_df(df, freqs="psd_freqs", psd="asd", hue="cond",
                xscale="log", yscale="log", xlabel="Frequency [Hz]",
                bands=None, title=None, ylabel=None, xlim=None, ylim=None,
                ax_kwargs={}, palette=None, save_name=None, col=None,
                legend=True, save_dir=cfg.FIG_RESULTS, show=False,
                col_order=None, hue_order=None, row=None, **rel_kwargs):
    msg = "You plot some PSDs twice!"
    if hue:
        n_base = df.groupby(["subject", "ch_hemisphere", hue])["subject"]
    else:
        n_base = df.groupby(["subject", "ch_hemisphere"])["subject"]
    assert n_base.nunique().unique()[0] == 1, msg
    group = df.groupby(["subject", "ch_hemisphere"])
    n_hemispheres = group.ngroup().unique()[-1] + 1
    proj_order = [proj for proj in cfg.PROJECT_ORDER
                  if proj in df.project.unique()]
    if hue == "cond" and hue_order is None and palette is None:
        df = df.set_index(hue).rename(index=cfg.COND_DICT).reset_index()
        df = df.reset_index().rename(columns={"cond": "LDOPA"})
        hue = "LDOPA"
        hue_order = ['off', 'on']
        palette = [cfg.COLOR_OFF, cfg.COLOR_ON]
    elif hue == "project" and hue_order is None and palette is None:
        hue_order = proj_order
        palette = [cfg.COLOR_DIC[proj] for proj in proj_order]
    if col == "project":
        col_order = proj_order
    elif col == "cond":
        df = df.set_index(col).rename(index=cfg.COND_DICT).reset_index()
        if set(df[col].unique()) == {'off', 'on'}:
            col_order = ["off", "on"]
        else:
            col_order = None
    elif col_order is not None:
        pass
    else:
        col_order = None
    if ylabel is None:
        if df.psd_kind.unique()[0] in ["cleaned", 'standard']:
            if psd == "asd":
                ylabel = r"ASD [$nV/\sqrt{Hz}$]"
            elif psd == "psd":
                ylabel = r"PSD [$\mu V^2/Hz$]"
        else:
            ylabel = "Normalized PSD [%]"
    if xlim:
        xmask = (df[freqs] >= xlim[0]) & (df[freqs] <= xlim[1])
        df = df.loc[xmask]
    g = sns.relplot(data=df, x=freqs, y=psd, hue=hue, hue_order=hue_order,
                    kind="line", palette=palette, col=col, col_order=col_order,
                    row=row, **rel_kwargs)
    g.set(xscale=xscale, yscale=yscale, **ax_kwargs,
          ylabel=ylabel, xlabel=xlabel, ylim=ylim, xlim=xlim)
    _add_band(bands, g)
    _add_band_annotations(bands, g)
    if not legend:
        g._legend.remove()
    for ax in g.axes.flatten():
        ax_title = ax.get_title()
        ax_title = ax_title.replace(f"{col} = ", "").replace(f"{row} = ", "")
        if col == "project":
            n_off, n_on, n_both = _get_sample_sizes(df, ax_title, hue, psd)
            sample_size_str = '\n('
            if n_off:
                sample_size_str += (r"$n_{off}=$"f"{n_off}, ")
            if n_on:
                sample_size_str += (r"$n_{on}=$"f"{n_on}, ")
            if n_both:
                sample_size_str += (r"$n_{both}=$"f"{n_both}")
            sample_size_str += ')'
            try:
                ax_title = cfg.PROJECT_DICT[ax_title]
            except KeyError:
                pass
            ax_title += sample_size_str
        ax.set_title(ax_title)
    if title == "n":
        g.fig.suptitle(f"n={n_hemispheres} hemispheres", fontsize=18, y=1.05)
    else:
        g.fig.suptitle(title, fontsize=18, y=1.05)
    close = not show
    if save_name:
        _save_fig(g, save_name, save_dir, transparent=True, close=close)
    else:
        plt.show()


def _get_sample_sizes(df, project, x, y):
    df = df[(df.project == project)]
    if not len(df):
        return np.nan, np.nan, np.nan
    # select only one frequency bin to count channels
    if 'psd_freqs' in df.columns:
        df = df[(df.psd_freqs == df.psd_freqs.values[0])]
    elif 'fm_freqs' in df.columns:
        df = df[(df.fm_freqs == df.fm_freqs.values[0])]
    n_off = len(df[(df.cond.isin(['off', 'off']))])
    n_on = len(df[(df.cond.isin(['on', 'ON']))])
    n_both = equalize_x_and_y(df, x, y)[1]
    return n_off, n_on, n_both


def _add_band(bands, g, alpha=.3, labels=False):
    if not bands:
        return None
    if isinstance(bands, str):
        colors = [cfg.BAND_COLORS[bands]]
        band_ranges = [cfg.BANDS[bands]]
        if labels:
            labels = [cfg.BAND_NAMES_GREEK[bands]]
        else:
            labels = [None]
    elif isinstance(bands, list):
        colors = [cfg.BAND_COLORS[band] for band in bands]
        band_ranges = [cfg.BANDS[band] for band in bands]
        if labels:
            labels = [cfg.BAND_NAMES_GREEK[band] for band in bands]
        else:
            labels = [None for _ in bands]
    for color, band_range, label in zip(colors, band_ranges, labels):
        band_start = band_range[0]
        band_len = np.diff(band_range)[0]
        if isinstance(g, sns.FacetGrid):
            for ax in g.axes.flatten():
                ylim = ax.get_ylim()
                rect = plt.Rectangle((band_start, ylim[0]), band_len, ylim[1],
                                     zorder=0,  # background
                                     label=label,
                                     color=color, alpha=alpha)
                ax.add_patch(rect)
        elif isinstance(g, plt.Axes):
            ylim = g.get_ylim()
            rect = plt.Rectangle((band_start, ylim[0]), band_len, ylim[1],
                                 zorder=0,  # background
                                 label=label,
                                 color=color, alpha=alpha)
            g.add_patch(rect)


def _add_band_annotations(bands, g, fontsize=5, short=False, y=1.05,
                          invisible=False):
    if not bands:
        return None
    band_dic = cfg.BAND_NAMES_GREEK_SHORT if short else cfg.BAND_NAMES_GREEK
    if isinstance(bands, str):
        bands = [bands]
    assert isinstance(bands, list)
    band_nmes = [band_dic[band] for band in bands]
    band_borders = [cfg.BANDS[band] for band in bands]
    alpha = 0 if invisible else 1
    for band_nme, band_border in zip(band_nmes, band_borders):
        band_start = band_border[0]
        band_len = np.diff(band_border)[0]
        ha = 'center'
        if isinstance(g, sns.FacetGrid):
            axes = g.axes.flatten()
        elif isinstance(g, plt.Axes):
            axes = [g]
        for ax in axes:
            axis_to_data = ax.transAxes + ax.transData.inverted()
            data_to_axis = axis_to_data.inverted()
            xmin, xmax = ax.get_xlim()
            if band_start + band_len > xmax:
                # avoid annotation outside of xlimits
                band_len = xmax - band_start
            cf = band_start + band_len / 2 if ha == 'center' else band_start
            cf_trans = data_to_axis.transform((cf, 0))[0]
            # va='top' is the only way to align mixture of greek and latin
            # letters correctly. Complicated data->axis conversion required
            # to enable annotation outside of ylimits.
            ax.annotate(band_nme, (cf_trans, y),
                        fontsize=fontsize, ha=ha,
                        xycoords='axes fraction',
                        color='grey', alpha=alpha,
                        va='top')


def _leg_titles(rhos, sample_sizes, title=None, corr_comparison=False,
                title_long=True):
    if corr_comparison and len(rhos) == 2:
        _, p_cond = independent_corr(*rhos, *sample_sizes)
        weight = "bold" if p_cond < 0.05 else "normal"
        if title_long:
            title = f"Condition: p={p_cond:.2f}"
        else:
            title = r'$p_{\text{off vs on}}$'f"={p_cond:.2f}"
        title_fontproperties = {'weight': weight}
    else:
        title_fontproperties = None
    return title, title_fontproperties


def channel_choice_histograms(df, save=True):
    """Plot histograms indicating which channels belong to which category."""
    save_dir = 'channel_choice_histograms'
    df = df[(df.psd_kind == 'standard') & (df.cond == 'off')
            & (df.subject != 'NeuEmptyroom')]

    hue_order = [proj for proj in cfg.PROJECT_ORDER
                 if proj in df.project.unique()]
    palette = [cfg.COLOR_DIC[proj] for proj in hue_order]
    hist_kwargs = dict(multiple='stack', hue_order=hue_order, palette=palette,
                       x="ch", hue="project", stat='count')

    # All channels
    _dataset_ch_distribution(df, hist_kwargs, save, save_dir)

    # Chs inside STN
    _dataset_chs_inside_STN(df, hist_kwargs, save, save_dir)

    # Sweetspot inside STN
    _dataset_sweetspot_inside_stn(df, hist_kwargs, save, save_dir)

    # Max. Beta Power
    _dataset_max_beta_chs(df, hist_kwargs, save, save_dir)

    # Directional leads
    _dataset_lead_directionality(df, hist_kwargs, save, save_dir)

    # DBS models
    _dataset_dbs_models(df[df.ch_choice], hist_kwargs, save_dir, save)


def _dataset_ch_distribution(df, hist_kwargs=None, save=None, save_dir=None):
    bip_chs = ['LFP_1-2', 'LFP_2-3', 'LFP_3-4']
    tan_chs = ['LFP_1-3', 'LFP_2-4']
    df_all = df[df.ch.isin(bip_chs + tan_chs) & ~df.ch_directional]
    df_all['ch'] = pd.Categorical(df_all['ch'], bip_chs + tan_chs)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.histplot(ax=ax, data=df_all, **hist_kwargs)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_title('All channels')
    plt.tight_layout()
    if save:
        fig_name = join(save_dir, 'all_channels')
        _save_fig(fig, fig_name, cfg.FIG_RESULTS)
    else:
        plt.show()


def _dataset_chs_inside_STN(df, hist_kwargs=None, save=None, save_dir=None):
    noMNI = df.ch_choice & df.mni_x.isna()
    bip_chs = ['LFP_1-2', 'LFP_2-3', 'LFP_3-4']
    mask = ~df.ch_bip_distant & df.ch.isin(bip_chs)
    inside_chs = mask & df.ch_inside_stn
    outside_chs = mask & ~df.ch_inside_stn & df.mni_x.notna()
    df_inside = df[inside_chs & ~df.ch_directional]
    df_outside = df[(outside_chs | noMNI) & ~df.ch_directional]
    df_outside.loc[noMNI, 'ch'] = 'MNI not\navailable'

    # sort chs for plotting
    df_inside['ch'] = pd.Categorical(df_inside['ch'], bip_chs)
    df_outside['ch'] = pd.Categorical(df_outside['ch'],
                                      bip_chs + ['MNI not\navailable'])

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.histplot(ax=ax[0], data=df_inside, **hist_kwargs)
    sns.histplot(ax=ax[1], data=df_outside, **hist_kwargs)
    ax[0].legend_.remove()
    sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
    ax[0].set_title('Inside STN')
    ax[1].set_title('Outside STN')
    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    # ax[1].set_yticks(None)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    plt.tight_layout()
    if save:
        fig_name = join(save_dir, 'inside_STN')
        _save_fig(fig, fig_name, cfg.FIG_RESULTS)
    else:
        plt.show()


def _dataset_sweetspot_inside_stn(df, hist_kwargs=None, save=None,
                                  save_dir=None):
    noMNI = df.ch_choice & df.mni_x.isna()
    df_sweet = df[(df.ch_sweetspot | noMNI) & ~df.ch_directional]
    mask = df.ch_sweetspot & ~df.ch_inside_stn
    df_sweet.loc[mask, 'ch'] = 'sweetspot\noutside STN'
    df_sweet.loc[noMNI, 'ch'] = 'MNI n/a'

    # sort chs for plotting
    bip_chs = ['LFP_1-2', 'LFP_2-3', 'LFP_3-4']
    ch_order = bip_chs + ['sweetspot\noutside STN', 'MNI n/a']
    df_sweet['ch'] = pd.Categorical(df_sweet['ch'], ch_order)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.histplot(ax=ax, data=df_sweet, **hist_kwargs)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_title('Sweet spot inside STN')
    plt.tight_layout()
    if save:
        fig_name = join(save_dir, 'sweetspot')
        _save_fig(fig, fig_name, cfg.FIG_RESULTS)
    else:
        plt.show()


def _dataset_max_beta_chs(df, hist_kwargs=None, save=None, save_dir=None):
    bip_chs = ['LFP_1-2', 'LFP_2-3', 'LFP_3-4']
    tan_chs = ['LFP_1-3', 'LFP_2-4']

    beta_standard = df.ch_beta_max & df.ch.isin(bip_chs)
    beta_tan = ((df.project == 'Tan') & df.ch_wiestpick
                & ~df.ch.isin(['LFP_WIEST']))
    df_beta = df[(beta_standard | beta_tan) & ~df.ch_directional]

    # sort chs for plotting
    df_beta['ch'] = pd.Categorical(df_beta['ch'], bip_chs + tan_chs)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.histplot(ax=ax, data=df_beta.dropna(subset='ch'), **hist_kwargs)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_title('Ch max. beta power')
    plt.tight_layout()
    if save:
        fig_name = join(save_dir, 'max_beta')
        _save_fig(fig, fig_name, cfg.FIG_RESULTS)
    else:
        plt.show()


def _dataset_lead_directionality(df, save=None, save_dir=None):
    df = df.drop_duplicates(subset=['subject'])

    df.loc[df.DBS_directional, 'ch'] = 'Directional'
    df.loc[~df.DBS_directional, 'ch'] = 'Non-directional'

    hue_order = [proj for proj in cfg.PROJECT_NAMES
                 if proj in df.project_nme.unique()]
    palette = [cfg.COLOR_DIC[proj] for proj in hue_order]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    sns.histplot(ax=ax, data=df, hue_order=hue_order, palette=palette,
                 hue="project_nme",
                 multiple='stack',
                 x="ch", stat='count')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=None)
    ax.set_title('Directional vs non-directional DBS leads')
    ax.set_xlabel('')
    plt.tight_layout()
    if save:
        fig_name = join(save_dir, 'directional_chs')
        _save_fig(fig, fig_name, cfg.FIG_RESULTS)
    else:
        plt.show()


def _dataset_dbs_models(df, save_dir=None, save=None):
    df = df.dropna(subset='DBS_model')
    df = df.drop_duplicates(subset=['subject'])

    dbs_model_dic = {
       'St. Jude Infinity directional':
           r'$\bf{St. Jude}$''\nInfinity directional',
       'Boston Scientific Vercise Cartesia':
           r'$\bf{Boston Scientific}$''\nVercise Cartesia',
       'Medtronic 3389':
           r'$\bf{Medtronic}$''\n3389',
       'Boston Scientific Vercise Standard':
           r'$\bf{Boston Scientific}$''\nVercise Standard',
       'St. Jude Infinity':
           r'$\bf{St. Jude}$''\nInfinity',
       'Medtronic SenSight Short':
           r'$\bf{Medtronic}$''\nSenSight Short',
       'Boston Scientific Vercise Cartesia X':
           r'$\bf{Boston Scientific}$''\nVercise Cartesia X',
       'Boston Scientific Vercise Cartesia HX':
           r'$\bf{Boston Scientific}$''\nVercise Cartesia HX'
    }
    df['DBS_model'] = df['DBS_model'].map(dbs_model_dic)

    hue_order = [proj for proj in cfg.PROJECT_NAMES
                 if proj in df.project_nme.unique()]
    palette = [cfg.COLOR_DIC[proj] for proj in hue_order]

    fig, ax = plt.subplots(1, 1, figsize=(3, .75))
    sns.histplot(ax=ax, data=df, hue_order=hue_order, palette=palette,
                 hue="project_nme", multiple='stack', x='DBS_model',
                 stat='count')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=None)
    ax.set_xlabel('')
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    plt.tight_layout()
    if save:
        if save_dir is None:
            save_dir = cfg.FIG_RESULTS
        _save_fig(fig, 'patients_DBSleads', save_dir, close=False)
    else:
        plt.show()


def _dataset_dbs_models_leads(df, save_dir=None, save=None, prefix=''):
    df = df.dropna(subset='DBS_model').copy()
    df = df.drop_duplicates(subset=['subject'])

    df.loc[df.DBS_directional, 'ch_dir'] = 'Yes'
    df.loc[~df.DBS_directional, 'ch_dir'] = 'No'

    dbs_model_dic = {
       'St. Jude Infinity directional':
           r'$\bf{St. Jude}$''\nInfinity directional',
       'Boston Scientific Vercise Cartesia':
           r'$\bf{Boston Scientific}$''\nVercise Cartesia',
       'Medtronic 3389':
           r'$\bf{Medtronic}$''\n3389',
       'Boston Scientific Vercise Standard':
           r'$\bf{Boston Scientific}$''\nVercise Standard',
       'St. Jude Infinity':
           r'$\bf{St. Jude}$''\nInfinity',
       'Medtronic SenSight Short':
           r'$\bf{Medtronic}$''\nSenSight Short',
       'Boston Scientific Vercise Cartesia X':
           r'$\bf{Boston Scientific}$''\nVercise Cartesia X',
       'Boston Scientific Vercise Cartesia HX':
           r'$\bf{Boston Scientific}$''\nVercise Cartesia HX'
    }
    df['DBS_model'] = df['DBS_model'].map(dbs_model_dic)

    hue_order = [proj for proj in cfg.PROJECT_NAMES
                 if proj in df.project_nme.unique()]
    palette = [cfg.COLOR_DIC[proj] for proj in hue_order]

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.5), width_ratios=[1, 6],
                             sharey=False)

    ax = axes[0]
    sns.histplot(ax=ax, data=df, x='ch_dir', hue='project_nme',
                palette=cfg.COLOR_DIC, multiple='stack',
                # legend=False,
                legend=True,
                shrink=0.8, hue_order=hue_order, stat='percent',
                common_norm=True, discrete=True)
    ax.set_xlabel('DBS directional')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylabel(None)
    ax.legend([], []).remove()
    ax = axes[1]
    sns.histplot(ax=ax, data=df, hue_order=hue_order, palette=palette,
                 hue="project_nme", multiple='stack', x='DBS_model',
                 legend=False,
                 stat='percent')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    plt.xticks(rotation=90, ha='center')
    sns.despine()
    handles = [plt.Line2D([0], [0], color=cfg.COLOR_DIC[proj], lw=4)
               for proj in hue_order]
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=cfg.COLOR_DIC[proj], label=proj)
               for proj in hue_order]
    labels = [proj for proj in hue_order]
    fig.legend(handles, labels, title=None, loc='lower left',
               bbox_to_anchor=(.0, -.01))
    plt.tight_layout()
    if save:
        if save_dir is None:
            save_dir = cfg.FIG_RESULTS
        _save_fig(fig, f'{prefix}DBSleads_models', save_dir, close=False,
                  bbox_inches=None)
    else:
        plt.show()


def _mni_coords_datasets(fig_dir=None, prefix=''):
    df = pd.read_excel(join(cfg.DF_PATH, 'localization_powers.xlsx'))
    # load dataframe with flipped mni coordinates
    msg = 'Run save_heatmaps.m first!'
    assert 'mni_xr' in df.columns, msg
    df = df[['subject', 'project', 'ch_nme', 'mni_xr', 'mni_yr', 'mni_zr']]
    df['project'] = df['project'].map(cfg.PROJECT_DICT)
    df['ch'] = df['ch_nme'].str.replace('_L_', '_').str.replace('_R_', '_')

    rename = {'mni_xr': 'mni X', 'mni_yr': 'mni Y', 'mni_zr': 'mni Z'}
    df.rename(columns=rename, inplace=True)
    values = rename.values()

    bip_chs = ['LFP_1-2', 'LFP_2-3', 'LFP_3-4']
    # bip_chs = ['LFP_1-3', 'LFP_2-4']  # dist channels don't exist in excel
    # sheet
    df = df[df.ch.isin(bip_chs)]
    # rename channels
    rename = {'LFP_1-2': '1-2', 'LFP_2-3': '2-3', 'LFP_3-4': '3-4'}
    df['ch'] = df['ch'].map({'LFP_1-2': '1-2', 'LFP_2-3': '2-3', 'LFP_3-4': '3-4'})
    order = [rename[ch] for ch in bip_chs]

    hue_order = [
        proj for proj in cfg.PROJECT_NAMES if proj in df.project.unique()
    ]
    palette = cfg.COLOR_DIC

    x = 'ch'
    # y = 'mni_xr'
    hue = 'project'

    pairs = [(('1-2', 'Berlin'), ('1-2', 'London')),
             (('1-2', 'Berlin'), ('1-2', 'Düsseldorf1')),
             (('1-2', 'London'), ('1-2', 'Düsseldorf1')),
             (('2-3', 'Berlin'), ('2-3', 'London')),
             (('2-3', 'Berlin'), ('2-3', 'Düsseldorf1')),
             (('2-3', 'London'), ('2-3', 'Düsseldorf1')),
             (('3-4', 'Berlin'), ('3-4', 'London')),
             (('3-4', 'Berlin'), ('3-4', 'Düsseldorf1')),
             (('3-4', 'London'), ('3-4', 'Düsseldorf1'))]
    # pairs=[(('LFP_1-2', 'Berlin'), ('LFP_1-2', 'London')),
    #        (('LFP_1-2', 'Berlin'), ('LFP_1-2', 'Düsseldorf1')),
    #        (('LFP_1-2', 'London'), ('LFP_1-2', 'Düsseldorf1')),
    #        (('LFP_2-3', 'Berlin'), ('LFP_2-3', 'London')),
    #        (('LFP_2-3', 'Berlin'), ('LFP_2-3', 'Düsseldorf1')),
    #        (('LFP_2-3', 'London'), ('LFP_2-3', 'Düsseldorf1')),
    #        (('LFP_3-4', 'Berlin'), ('LFP_3-4', 'London')),
    #        (('LFP_3-4', 'Berlin'), ('LFP_3-4', 'Düsseldorf1')),
    #        (('LFP_3-4', 'London'), ('LFP_3-4', 'Düsseldorf1'))]

    # pairs = [
    #     (('LFP_1-3', 'Berlin'), ('LFP_1-3', 'London')),
    #     (('LFP_1-3', 'Berlin'), ('LFP_1-3', 'Düsseldorf1')),
    #     (('LFP_1-3', 'London'), ('LFP_1-3', 'Düsseldorf1')),
    #     (('LFP_2-4', 'Berlin'), ('LFP_2-4', 'London')),
    #     (('LFP_2-4', 'Berlin'), ('LFP_2-4', 'Düsseldorf1')),
    #     (('LFP_2-4', 'London'), ('LFP_2-4', 'Düsseldorf1'))
    # ]

    stat_params = dict(test='Mann-Whitney', text_format='star', loc='outside',
                       verbose=False, show_test_name=False, line_width=0.3)
    params = dict(data=df, x=x, order=order, hue=hue, hue_order=hue_order,
                  fliersize=0.1, saturation=1, linewidth=0.2)

    fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.6), sharex=True)

    for i, y in enumerate(values):
        ax = axes[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.boxplot(ax=ax, palette=palette, y=y, **params)

        annotator = Annotator(ax=ax, pairs=pairs, y=y, **params)
        annotator.configure(**stat_params)
        annotator.apply_and_annotate()
        ax.set_xlabel(None)
        ax.legend_.remove()
    # fig.supxlabel('DBS channel')
    sns.despine(bottom=True)
    plt.tight_layout()
    _save_fig(fig, f'{prefix}dataset_mniCoords', join(cfg.FIG_PAPER, fig_dir),
              close=False, bbox_inches=None)


def _dataset_comparison(df, save_dir=None, save=None):
    # Prepare df
    df = df.copy()
    hue_order = [proj for proj in cfg.PROJECT_NAMES
                 if proj in df.project_nme.unique()]

    # cond order
    conds = ['off', 'on', 'offon_abs']
    df['cond'] = pd.Categorical(df['cond'], conds)
    rename = cfg.COND_DICT.copy()
    rename['offon_abs'] = 'off&on'
    df['cond'] = df['cond'].map(rename)

    fig, axes = plt.subplots(1, 9, figsize=(7, .85),
                             width_ratios=[1.5, 1, 2, 1, 1, 1, 1, 1, 1])

    # Sample sizes subject ####################################################
    ax = axes[0]

    duplicates = ['project', 'subject', 'cond']
    df_sub_cond = df.drop_duplicates(subset=duplicates)

    kwargs = dict(hue='project_nme', palette=cfg.COLOR_DIC,
                  multiple='dodge', x='cond', hue_order=hue_order,
                  stat='count', legend=False, shrink=0.8)
    sns.histplot(ax=ax, data=df_sub_cond, **kwargs)
    ax.set_xlabel('Levodopa')
    ax.set_ylabel(cfg.SAMPLE_PAT)
    ###########################################################################

    # # Sample sizes hemispheres  ###############################################
    # ax = axes[1]

    # df_hemi = df.drop_duplicates(subset=duplicates + ['ch_hemisphere'])
    # sns.histplot(ax=ax, data=df_hemi, **kwargs)
    # # ax.set_ylabel(r'$n_{\mathrm{hemi}}$')
    # # ax.set_xlabel('# Hemispheres')
    # ax.set_ylabel(None)
    # ax.set_xlabel(r'$n_{\mathrm{hemispheres}}$')
    # ###########################################################################

    # UPDRS pre-post operative  ###############################################
    ax = axes[1]
    df_sub = df.drop_duplicates(subset=['project', 'subject'])
    df_sub = df_sub[df_sub.project != 'all']
    df_sub['UPDRS_preOp'] = df_sub.UPDRS_pre_III.notna()
    df_sub['UPDRS_postOp'] = df_sub.UPDRS_post_III.notna()
    df_long = pd.melt(df_sub,
                    id_vars='project_nme',
                    value_vars=['UPDRS_preOp', 'UPDRS_postOp'],
                    var_name='prepost',
                    value_name='has_score')
    # Filter only rows where a score exists
    df_long = df_long[df_long['has_score']]
    kwargs = dict(hue='project_nme', palette=cfg.COLOR_DIC,
                  multiple='stack', x='prepost', hue_order=hue_order,
                  stat='count', legend=False, shrink=0.8)
    sns.histplot(ax=ax, data=df_long, **kwargs)
    ax.set_xticklabels(['Pre', 'Post'])
    ax.set_ylabel(None)
    # ax.set_xlabel('UPDRS assessment')
    ax.set_xlabel('Pre/post surgery')
    ###########################################################################

    # Recording  ##############################################################
    ax = axes[2]

    sns.histplot(ax=ax, data=df_sub, x='patient_days_after_implantation',
                 hue='project_nme',
                 palette=cfg.COLOR_DIC, multiple='stack', legend=False,
                 shrink=0.8, hue_order=hue_order, stat='count',
                 common_norm=True, discrete=True)
    xticks = range(0, 8)
    ax.set_xticks(xticks)
    ax.set_xlabel('Days after surgery')
    ax.set_ylabel(None)
    ###########################################################################

    # Sex  ####################################################################
    ax = axes[3]

    # don't plot unknown sex because not informative
    df_sub_gender = df_sub[df_sub.patient_sex.isin(['male', 'female'])]
    df_sub_gender['patient_sex'] = df_sub_gender['patient_sex'].map(
        {'male': 'M', 'female': 'F'}
    )

    kwargs = dict(hue='project_nme', palette=cfg.COLOR_DIC,
                  legend=False,
                  common_norm=False,  # True overemphasizes 'all'
                  common_grid=True, bw_method=0.5, cut=0.1)
    # columns without variance cannot be plotted with kdeplot
    sns.histplot(ax=ax, data=df_sub_gender, x='patient_sex', hue='project_nme',
                    palette=cfg.COLOR_DIC,
                    multiple='dodge', legend=False,
                    shrink=0.8,
                    hue_order=hue_order,
                    stat='percent',
                    common_norm=False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel(None)
    ax.set_xlabel('Sex')
    ###########################################################################

    # Age #####################################################################
    ax = axes[4]
    sns.kdeplot(ax=ax, data=df_sub, x='patient_age', **kwargs)
    ax.set_xlabel('Age [yrs]')
    ax.set_ylabel('Density')
    ax.set_yticks([])
    ###########################################################################

    # Disease duration ########################################################
    ax = axes[5]
    sns.kdeplot(ax=ax, data=df_sub, x='patient_disease_duration', **kwargs)
    ax.set_xlabel('PD duration [yrs]')
    ax.set_yticks([])
    ax.set_ylabel(None)
    ###########################################################################

    # Symptoms  ###############################################################
    df_updrs = df.drop_duplicates(subset=['project', 'subject', 'cond',
                                          'ch_hemisphere'])
    # remove pooled data from UPDRS scores -> confusing because too many lines
    # and colors and no pooled data for other graphs
    df_updrs = df_updrs[df_updrs.project != 'all']

    kwargs = dict(hue='project_nme', palette=cfg.COLOR_DIC, common_norm=False,
                  hue_order=hue_order, legend=False,
                  common_grid=True, bw_method=0.5, cut=0.1)
    symptoms = ['UPDRS_bradyrigid_contra', 'UPDRS_tremor_contra', 'UPDRS_III']
    xlabels = ['Bradykinesia-rigidity', 'Tremor', 'Total UPDRS-III']
    symptoms = symptoms[::-1]
    xlabels = xlabels[::-1]
    for j, cond in enumerate(['off']):
        for i, symptom in enumerate(symptoms):
            no_tremor = ['Tan', 'Litvak'] if 'tremor' in symptom else []
            mask = (df_updrs.cond == cond) & ~df_updrs.project.isin(no_tremor)
            ax = axes[-(i+1)]
            sns.kdeplot(ax=ax, data=df_updrs[mask], x=symptom, **kwargs)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(
                xmax=1, decimals=0)
            )
            ax.set_xlabel(xlabels[i])
            ax.set_yticks([])
    ###########################################################################

    plt.tight_layout()
    if save:
        if save_dir is None:
            save_dir = join(cfg.FIG_RESULTS, 'patients', 'DBSleads')
        _save_fig(fig, 'multicenter_comparison', save_dir, close=False,
                  transparent=True, bbox_inches=None)
    else:
        plt.show()


def _dataset_comparison_divided(df, save_dir=None, save=None, prefix=''):
    # Prepare df
    df = df.copy()
    hue_order = [proj for proj in cfg.PROJECT_NAMES
                 if proj in df.project_nme.unique()]

    # cond order
    conds = ['off', 'on', 'offon_abs']
    df['cond'] = pd.Categorical(df['cond'], conds)
    rename = cfg.COND_DICT.copy()
    rename['offon_abs'] = 'off&on'
    df['cond'] = df['cond'].map(rename)

    fig, axes = plt.subplots(1, 3, figsize=(2.7, .85), width_ratios=[1.5, 1, 2])

    # Sample sizes subject ####################################################
    ax = axes[0]

    duplicates = ['project', 'subject', 'cond']
    df_sub_cond = df.drop_duplicates(subset=duplicates)

    kwargs = dict(hue='project_nme', palette=cfg.COLOR_DIC,
                  multiple='dodge', x='cond', hue_order=hue_order,
                  stat='count', legend=False, shrink=0.8)
    sns.histplot(ax=ax, data=df_sub_cond, **kwargs)
    ax.set_xlabel('Levodopa')
    ax.set_ylabel(cfg.SAMPLE_PAT)
    ###########################################################################

    # UPDRS pre-post operative  ###############################################
    ax = axes[1]
    df_sub = df.drop_duplicates(subset=['project', 'subject'])
    df_sub = df_sub[df_sub.project != 'all']
    df_sub['UPDRS_preOp'] = df_sub.UPDRS_pre_III.notna()
    df_sub['UPDRS_postOp'] = df_sub.UPDRS_post_III.notna()
    df_long = pd.melt(df_sub,
                      id_vars='project_nme',
                      value_vars=['UPDRS_preOp', 'UPDRS_postOp'],
                      var_name='prepost',
                      value_name='has_score')
    # Filter only rows where a score exists
    df_long = df_long[df_long['has_score']]
    kwargs = dict(hue='project_nme', palette=cfg.COLOR_DIC,
                  multiple='stack', x='prepost', hue_order=hue_order,
                  stat='count', legend=False, shrink=0.8)
    sns.histplot(ax=ax, data=df_long, **kwargs)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre', 'Post'])
    ax.set_ylabel(None)
    ax.set_xlabel('Pre/post surgery')
    ###########################################################################

    # Recording  ##############################################################
    ax = axes[2]

    sns.histplot(ax=ax, data=df_sub, x='patient_days_after_implantation',
                 hue='project_nme',
                 palette=cfg.COLOR_DIC, multiple='stack', legend=False,
                 shrink=0.8, hue_order=hue_order, stat='count',
                 common_norm=True, discrete=True)
    xticks = range(0, 8)
    ax.set_xticks(xticks)
    ax.set_xlabel('Days after surgery')
    ax.set_ylabel(None)
    ###########################################################################

    plt.tight_layout()
    if save:
        if save_dir is None:
            save_dir = join(cfg.FIG_RESULTS, 'patients', 'DBSleads')
        _save_fig(fig, 'D1__multicenter_comparison', save_dir, close=False,
                  transparent=True, bbox_inches=None)
    else:
        plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(2.1, .85))

    # Sex  ####################################################################
    ax = axes[0]

    # don't plot unknown sex because not informative
    df_sub_gender = df_sub[df_sub.patient_sex.isin(['male', 'female'])]
    df_sub_gender.loc[:, 'patient_sex'] = df_sub_gender['patient_sex'].map(
        {'male': 'M', 'female': 'F'}
    )

    kwargs = dict(hue='project_nme', palette=cfg.COLOR_DIC,
                  legend=False,
                  common_norm=False,  # True overemphasizes 'all'
                  common_grid=True, bw_method=0.5, cut=0.1)
    # columns without variance cannot be plotted with kdeplot
    sns.histplot(ax=ax, data=df_sub_gender, x='patient_sex', hue='project_nme',
                    palette=cfg.COLOR_DIC,
                    multiple='dodge', legend=False,
                    shrink=0.8,
                    hue_order=hue_order,
                    stat='percent',
                    common_norm=False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel(None)
    ax.set_xlabel('Sex')
    ###########################################################################

    # Age #####################################################################
    ax = axes[1]
    sns.kdeplot(ax=ax, data=df_sub, x='patient_age', **kwargs)
    ax.set_xlabel('Age [yrs]')
    ax.set_ylabel('Density')
    ax.set_yticks([])
    ###########################################################################

    # Disease duration ########################################################
    ax = axes[2]
    sns.kdeplot(ax=ax, data=df_sub, x='patient_disease_duration', **kwargs)
    ax.set_xlabel('PD duration [yrs]')
    ax.set_yticks([])
    ax.set_ylabel(None)
    ###########################################################################
    plt.tight_layout()
    if save:
        if save_dir is None:
            save_dir = join(cfg.FIG_RESULTS, 'patients', 'DBSleads')
        _save_fig(fig, 'D2__multicenter_comparison', save_dir, close=False,
                  transparent=True, bbox_inches=None)
    else:
        plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(2.1, .85))
    # Symptoms  ###############################################################
    df_updrs = df.drop_duplicates(subset=['project', 'subject', 'cond',
                                          'ch_hemisphere'])
    # remove pooled data from UPDRS scores -> confusing because too many lines
    # and colors and no pooled data for other graphs
    df_updrs = df_updrs[df_updrs.project != 'all']

    kwargs = dict(hue='project_nme', palette=cfg.COLOR_DIC, common_norm=False,
                  hue_order=hue_order, legend=False,
                  common_grid=True, bw_method=0.5, cut=0.1)
    symptoms = ['UPDRS_bradyrigid_contra', 'UPDRS_tremor_contra', 'UPDRS_III']
    xlabels = ['Bradykinesia-rigidity', 'Tremor', 'Total UPDRS-III']
    symptoms = symptoms[::-1]
    xlabels = xlabels[::-1]
    # symptoms = ['UPDRS_bradyrigid_contra', 'UPDRS_tremor_contra', 'UPDRS_III']
    # xlabels = ['Bradykinesia-rigidity', 'Tremor', 'Total UPDRS-III']
    for j, cond in enumerate(['off']):
        for i, symptom in enumerate(symptoms):
            no_tremor = ['Tan', 'Litvak'] if 'tremor' in symptom else []
            mask = (df_updrs.cond == cond) & ~df_updrs.project.isin(no_tremor)
            ax = axes[-(i+1)]
            sns.kdeplot(ax=ax, data=df_updrs[mask], x=symptom, **kwargs)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(
                xmax=1, decimals=0
            ))
            ax.set_xlabel(xlabels[i])
            ax.set_yticks([])
    ###########################################################################

    plt.tight_layout()
    if save:
        if save_dir is None:
            save_dir = join(cfg.FIG_RESULTS, 'patients', 'DBSleads')
        _save_fig(fig, f'D3__multicenter_comparison', save_dir, close=False,
                  transparent=True, bbox_inches=None)
    else:
        plt.show()


def _dataset_overview(df_n, fig_dir=None, prefix=''):
    mask = (df_n.UPDRS_exists & df_n.has_model)
    mask_off = (mask & df_n.asymmetric_off & df_n.both_hemis_off_available)
    mask_on = (mask & df_n.asymmetric_on & df_n.both_hemis_on_available
               & df_n.dominant_side_consistent)
    # sort projects in df_n according to cfg.PROJECT_NAMES
    df_n['project_nme'] = pd.Categorical(
        df_n['project_nme'], categories=cfg.PROJECT_NAMES, ordered=True
    )

    fig, ax = plt.subplots(1, 1, figsize=(1.55, 1.5), sharey=True)

    # Full data
    sns.histplot(
        data=df_n, x='project_nme', discrete=True, stat="count", ax=ax,
        label='Original'
    )

    # Iterate through each bar and set the color based on the project name
    for i, patch in enumerate(ax.patches):
        project = cfg.PROJECT_NAMES[i]
        patch.set_facecolor(cfg.COLOR_DIC[project])
        patch.set_alpha(0.2)

    # Filtered data Off
    sns.histplot(
        data=df_n[mask_off], x='project_nme', discrete=True, stat="count",
        ax=ax, label=f'Asymmetric {cfg.COND_DICT["off"]}'
    )
    for i, patch in enumerate(ax.patches[5:]):
        project = cfg.PROJECT_NAMES[i]
        patch.set_facecolor(cfg.COLOR_DIC[project])
        patch.set_alpha(0.4)


    # Filtered data On
    sns.histplot(
        data=df_n[mask_on], x='project_nme', discrete=True, stat="count",
        ax=ax, label=f'Asymmetric {cfg.COND_DICT["on"]}'
    )
    for i, patch in enumerate(ax.patches[10:]):
        project = cfg.PROJECT_NAMES[i]
        patch.set_facecolor(cfg.COLOR_DIC[project])

    ax.legend()
    ax.set_xlabel(None)
    ax.set_xticks(range(len(cfg.PROJECT_NAMES)))
    ax.set_xticklabels(cfg.PROJECT_NAMES, rotation=40, ha='right')
    ax.tick_params(axis='x', pad=0.5)
    ax.set_ylabel(r'$n_{\text{sub}}$')

    plt.tight_layout()
    fpath = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, f'{prefix}patients_sample_size_asymmetric', fpath,
              close=False, bbox_inches=None)


def correlations_offon(df, redundancies=[], save_name=None):
    save_dir = join('correlations', save_name)

    # rename columns to make understandable
    df = df.reindex(sorted(df.columns, key=str.lower), axis=1)
    df = df.rename(columns=cfg.PLOT_LABELS)

    df_off = df[df.cond == 'off']
    df_on = df[df.cond == 'on']
    df_offon = df[df.cond == 'offon_abs']

    # calculate correlations
    kwargs = dict(method='spearman', numeric_only=True, min_periods=3)
    df_corr_off = df_off.corr(**kwargs)
    df_corr_on = df_on.corr(**kwargs)
    df_corr_offon = df_offon.corr(**kwargs)

    # calculate p-values
    kwargs['method'] = p_value_df("spearman", "spearman")
    df_pval_off = df_off.corr(**kwargs)
    df_pval_on = df_on.corr(**kwargs)
    df_pval_offon = df_corr_offon.corr(**kwargs)

    # calculate sample sizes
    kwargs['method'] = sample_size_df()
    df_n_off = df_off.corr(**kwargs)
    df_n_on = df_on.corr(**kwargs)
    df_n_offon = df_corr_offon.corr(**kwargs)

    # combine ON and OFF in single matrix - make the lower triangle OFF and
    # the upper triangle ON. Set upper ONOFF triangle to nan.
    df_corr = df_corr_off
    df_pval = df_pval_off
    df_n = df_n_off

    upper_indices = np.triu_indices_from(df_corr)
    df_corr.values[upper_indices] = df_corr_on.values[upper_indices]
    df_pval.values[upper_indices] = df_pval_on.values[upper_indices]
    df_n.values[upper_indices] = df_n_on.values[upper_indices]

    df_corr_offon.values[upper_indices] = np.nan
    df_pval_offon.values[upper_indices] = np.nan
    df_n_offon.values[upper_indices] = np.nan

    # set diagonal elements to nan
    np.diag_indices = np.diag_indices_from(df_corr)
    df_corr.values[np.diag_indices] = np.nan
    df_pval.values[np.diag_indices] = np.nan
    df_n.values[np.diag_indices] = np.nan

    df_corr_offon.values[np.diag_indices] = np.nan
    df_pval_offon.values[np.diag_indices] = np.nan
    df_n_offon.values[np.diag_indices] = np.nan

    # set redundant correlations to nan
    for redundant in redundancies:
        redundant_indices = df_corr.index.str.contains(redundant)
        redundant_indices = (
            redundant_indices[:, None] * redundant_indices[None]
        )
        df_corr.values[redundant_indices] = np.nan
        df_pval.values[redundant_indices] = np.nan
        df_n.values[redundant_indices] = np.nan

        df_corr_offon.values[redundant_indices] = np.nan
        df_pval_offon.values[redundant_indices] = np.nan
        df_n_offon.values[redundant_indices] = np.nan

    # discriminate between significant and non-significant correlations
    psig = 0.05
    mask_sig = df_pval < psig
    mask_unsig = df_pval >= psig
    df_corr_sig = df_corr[mask_sig]
    df_corr_unsig = df_corr[mask_unsig]

    mask_sig_offon = df_pval_offon < psig
    mask_unsig_offon = df_pval_offon >= psig
    df_corr_offon_sig = df_corr_offon[mask_sig_offon]
    df_corr_offon_unsig = df_corr_offon[mask_unsig_offon]

    # make annotation matrix
    df_annot = ('r=' + df_corr.round(2).astype(str)
                + "\np=" + df_pval.round(2).astype(str)
                + "\nn=" + df_n.astype('Int64').astype(str))
    df_annot_offon = ('r=' + df_corr_offon.round(2).astype(str)
                      + "\np=" + df_pval_offon.round(2).astype(str)
                      + "\nn=" + df_n_offon.astype('Int64').astype(str))
    df_annot = df_annot.to_numpy()
    df_annot_offon = df_annot_offon.to_numpy()

    cmap_sig = 'seismic'
    cmap_unsig = sns.color_palette("Greys", n_colors=1, desat=1)
    heat_kwargs = dict(linewidth=.5, square=True, fmt='', vmin=-1, vmax=1)
    size_annot = 35 / np.sqrt(len(df_corr))
    annot_kws = dict(size=size_annot)

    fig, axes = plt.subplots(1, 3, figsize=(40, 20),
                             gridspec_kw={'width_ratios': [1, 1, 0.08]})
    ax = axes[0]
    sns.heatmap(df_corr_sig, ax=ax, annot=df_annot, cmap=cmap_sig,
                cbar=False, **heat_kwargs, annot_kws=annot_kws)
    sns.heatmap(df_corr_unsig, ax=ax, annot=df_annot, cmap=cmap_unsig,
                cbar=False, **heat_kwargs, annot_kws=annot_kws)
    ax.set_xlabel('off', fontsize=40, x=0, horizontalalignment='left')
    ax.set_title('ON', fontsize=40, x=1, horizontalalignment='right')

    ax = axes[1]
    sns.heatmap(df_corr_offon_sig, annot=df_annot_offon, cbar_ax=axes[2],
                cbar_kws={'label': 'Spearman correlation coefficient'},
                ax=ax, cmap=cmap_sig, **heat_kwargs, annot_kws=annot_kws)
    sns.heatmap(df_corr_offon_unsig, annot=df_annot_offon,
                ax=ax, cmap=cmap_unsig, cbar=False, **heat_kwargs,
                annot_kws=annot_kws)
    ax.set_yticklabels([])
    ax.set_xlabel('OFF-ON', fontsize=40, x=0, horizontalalignment='left')
    plt.tight_layout()
    _save_fig(fig, save_dir, cfg.FIG_RESULTS)


def band_power_channel_choice(df, save=True):
    """Attention: I add as many channels as possible. When MNI are missing for
    Neumann, I will not add them to sweetspot but I will add them to max. beta.
    This means, the cohorts used for the comparisons are not identical.

    I can remove them to make effect sizes more comparable. However, this will
    result in less power for the other comparisons."""
    save_dir = join('band_powers', 'robustness')
    df = df[df.cond.isin(['on', 'off']) & (df.psd_kind == 'standard')]
    bands = ["beta_low"]

    max_beta = df.ch_beta_max & (~df.ch_bip_distant | df.project.isin(['Tan']))
    sweet_inside = (df.ch_sweetspot & df.ch_inside_stn)
    ch_selections = [('Max. Beta (adj.)', max_beta),
                     ('Sweetspot inside STN (adj.)', sweet_inside),
                     ('PSD adj. ch mean inside STN', df.ch_mean_inside_stn),
                     ('Mean of adj. chs inside STN', df.ch_inside_stn_mean),
                     ('LFP_1-2 (adj.)', (df.ch == 'LFP_1-2')),
                     ('LFP_2-3 (adj.)', (df.ch == 'LFP_2-3')),
                     ('LFP_3-4 (adj.)', (df.ch == 'LFP_3-4')),
                     ('LFP_1-3', (df.ch == 'LFP_1-3')),
                     ('LFP_2-4', (df.ch == 'LFP_2-4')),
                     ('All adj. chs mean', df.ch_mean)]
    hue = 'cond'
    for band in bands:
        y = f"{band}_abs_max_log"
        if band == 'HFO':
            y += "_nonoise"
        save_path = join(save_dir, f'band_powers_robustness_chs_{band}')

        ch_descriptions = []
        project_list = []
        effect_sizes = []
        colors = []
        pvals = []
        sample_sizes_wilcoxon = []
        sample_sizes_cohen = []
        annotations = []

        projects = df.project.unique()
        projects = [proj for proj in cfg.PROJECT_ORDER if proj in projects]
        proj_slices = [df.project == col_val for col_val in projects]
        # summarize all projects with mni coordinates
        projects.append('all')
        proj_slices.append(df.project.isin(['Neumann', 'Florin', 'Litvak']))
        for ch_selection, mask in ch_selections:
            df_mask = df[mask]
            for idx, proj in enumerate(projects):
                df_sub = df_mask[proj_slices[idx]]
                effect_size, n_d = _cohen_stats(df_sub, y, hue)
                pval, n_wil = _wilcoxon_stats(df_sub, y, hue, ch_selection,
                                              mask)
                effect_str = f"d={effect_size:.2f}\n(n={n_d})"
                if abs(pval) < 1e-2:
                    pval = f"p={pval:1.0e}"
                else:
                    pval = f"p={pval:.2f}"
                wilcoxon_str = f"\n{pval}\n(n={n_wil})"
                annotation = effect_str + wilcoxon_str
                color = cfg.COLOR_DIC[proj]
                if proj == 'all':
                    project_str = 'Ber+Dü1+Lon'
                else:
                    project_str = cfg.PROJECT_DICT[proj]

                effect_sizes.append(effect_size)
                sample_sizes_cohen.append(n_d)
                pvals.append(pval)
                sample_sizes_wilcoxon.append(n_wil)
                project_list.append(project_str)
                colors.append(color)
                ch_descriptions.append(ch_selection)
                annotations.append(annotation)

        stats = pd.DataFrame({'effect_sizes': effect_sizes,
                              'pvals': pvals,
                              'sample_sizes_wilcoxon': sample_sizes_wilcoxon,
                              'sample_sizes_cohen': sample_sizes_cohen,
                              'project': project_list,
                              'colors': colors,
                              'ch_descriptions': ch_descriptions,
                              'annotations': annotations})

        # split in two rows
        descrs = stats.ch_descriptions.unique().tolist()
        ch_descriptions1 = descrs[:len(descrs) // 2]
        ch_descriptions2 = descrs[len(descrs) // 2:]
        stats1 = stats[stats.ch_descriptions.isin(ch_descriptions1)]
        stats2 = stats[stats.ch_descriptions.isin(ch_descriptions2)]

        fig, axes = plt.subplots(2, 1, figsize=(3*len(ch_descriptions2), 10),
                                 sharey=True)
        for ax, stats in zip(axes, [stats1, stats2]):
            sns.barplot(ax=ax, x='ch_descriptions', y='effect_sizes',
                        data=stats, hue='project', palette=colors)
            for patch in ax.patches:
                width = patch.get_width()
                # make bar for combined 'all' projects wider
                if patch._facecolor == (0, 0, 0, 1):
                    patch.set_width(width*2)
            for idx, proj in enumerate(stats.project.unique()):
                label_type = 'edge' if band == 'HFO' else 'center'
                color = 'k' if band == 'HFO' else 'white'
                if proj == 'Ber+Dü1+Lon':
                    labels = stats[(stats.project == proj)]
                    labels = labels.annotations.to_list()
                    ax.bar_label(ax.containers[idx], labels=labels,
                                 fontsize=10, label_type=label_type,
                                 color=color, padding=0.5)
            ax.set_ylabel('Cohen\'s d')

        axes[0].legend().remove()
        sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1))
        axes[0].set_xlabel('')
        axes[1].set_xlabel('Channel selection')
        plt.suptitle(f'Channel choice robustness {band}')
        plt.tight_layout()
        if save:
            _save_fig(fig, save_path, cfg.FIG_RESULTS)
        else:
            plt.show()


def power_definitions_barplot(df):
    save_dir = join('band_powers', 'robustness')
    df = df[df.cond.isin(['on', 'off']) & (df.psd_kind == 'standard')
            & df.ch_choice]
    hue = 'cond'
    pwr_definitions = [('Peak Max.', '_abs_max_log'),
                       ('Peak Max. +- 1 Hz', '_abs_max_3Hz_log'),
                       ('Peak Max. +- 3 Hz', '_abs_max_7Hz_log'),
                       ('Band Mean', '_abs_mean_log')]
    band = 'beta_low'
    save_path = join(save_dir, f'band_powers_robustness_defs_{band}')
    pwr_descriptions = []
    project_list = []
    effect_sizes = []
    colors = []
    pvals = []
    sample_sizes_wilcoxon = []
    sample_sizes_cohen = []
    annotations = []

    projects = df.project.unique()
    projects = [proj for proj in cfg.PROJECT_ORDER if proj in projects]
    proj_slices = [df.project == col_val for col_val in projects]
    projects.append('all')
    proj_slices.append(slice(None))
    for pwr_descr, col in pwr_definitions:
        y = band + col
        if band == 'HFO':
            y += "_nonoise"
        for idx, proj in enumerate(projects):
            df_sub = df[proj_slices[idx]]
            effect_size, n_d = _cohen_stats(df_sub, y, hue)
            pval, n_wil = _wilcoxon_stats(df_sub, y, hue)
            effect_str = f"d={effect_size:.2f}\n(n={n_d})"
            pval = f"p={pval:1.0e}" if abs(pval) < 1e-2 else f"p={pval:.2f}"
            wilcoxon_str = f"\n{pval}\n(n={n_wil})"
            annotation = effect_str + wilcoxon_str

            effect_sizes.append(effect_size)
            sample_sizes_cohen.append(n_d)
            pvals.append(pval)
            sample_sizes_wilcoxon.append(n_wil)
            project_list.append(cfg.PROJECT_DICT[proj])
            colors.append(cfg.COLOR_DIC[proj])
            pwr_descriptions.append(pwr_descr)
            annotations.append(annotation)

    stats = pd.DataFrame({'effect_sizes': effect_sizes,
                          'pvals': pvals,
                          'sample_sizes_wilcoxon': sample_sizes_wilcoxon,
                          'sample_sizes_cohen': sample_sizes_cohen,
                          'project': project_list,
                          'colors': colors,
                          'pwr_descriptions': pwr_descriptions,
                          'annotations': annotations})

    fig, axes = plt.subplots(1, 1, figsize=(20, 5))
    sns.barplot(ax=axes, x='pwr_descriptions', y='effect_sizes', data=stats,
                hue='project', palette=colors)
    for patch in axes.patches:
        width = patch.get_width()
        # make bar for combined 'all' projects wider
        if patch._facecolor == (0, 0, 0, 1):
            patch.set_width(width*2)
    for idx, proj in enumerate(stats.project.unique()):
        if proj != 'all':
            continue
        labels = stats[(stats.project == proj)].annotations.to_list()
        axes.bar_label(axes.containers[idx], labels=labels, fontsize=10,
                       label_type='center', color='white')
    sns.move_legend(axes, "upper left", bbox_to_anchor=(1, 1))
    axes.set_ylabel('Cohen\'s d')
    axes.set_xlabel('Power definition')
    plt.suptitle(f'Power definition robustness {band}')
    plt.tight_layout()
    _save_fig(fig, save_path, cfg.FIG_RESULTS)


# def dbslead_directionality_barplot(df):
#     """Test robustness of effect sizes with regard to DBS leads.

#     Tests:
#     - Directional vs non-directional leads all manufacturers
#     - BS vs MT vs SJ for comparable leads
#     - any other important options?"""

#     save_dir = join('band_powers', 'robustness')
#     df = df[df.cond.isin(['on', 'off']) & (df.psd_kind == 'standard')
#             & df.ch_choice]
#     hue = 'cond'
#     dbs_directional = [('DBS Directional', df.DBS_directional),
#                        ('DBS Non-Directional', ~df.DBS_directional)]
#     bands = ['beta_low', 'HFO']
#     for band in bands:
#         save_path = join(save_dir, f'band_powers_robustness_lead_dir_{band}')
#         lead_descriptions = []
#         project_list = []
#         effect_sizes = []
#         colors = []
#         pvals = []
#         sample_sizes_wilcoxon = []
#         sample_sizes_cohen = []
#         annotations = []

#         projects = df.project.unique()
#         projects = [proj for proj in cfg.PROJECT_ORDER if proj in projects]
#         proj_slices = [df.project == col_val for col_val in projects]
#         projects.append('all')
#         proj_slices.append(slice(None))
#         for descr, mask in dbs_directional:
#             y = band + '_abs_max_log'
#             if band == 'HFO':
#                 y += "_nonoise"
#             for idx, proj in enumerate(projects):
#                 df_sub = df[proj_slices[idx]]
#                 df_sub = df_sub[mask]
#                 effect_size, n_d = _cohen_stats(df_sub, y, hue)
#                 pval, n_wil = _wilcoxon_stats(df_sub, y, hue)
#                 effect_str = f"d={effect_size:.2f}\n(n={n_d})"
#                 pval = f"p={pval:1.0e}" if abs(pval) < 1e-2 else f"p={pval:.2f}"
#                 wilcoxon_str = f"\n{pval}\n(n={n_wil})"
#                 annotation = effect_str + wilcoxon_str

#                 effect_sizes.append(effect_size)
#                 sample_sizes_cohen.append(n_d)
#                 pvals.append(pval)
#                 sample_sizes_wilcoxon.append(n_wil)
#                 project_list.append(cfg.PROJECT_DICT[proj])
#                 colors.append(cfg.COLOR_DIC[proj])
#                 lead_descriptions.append(descr)
#                 annotations.append(annotation)

#         stats = pd.DataFrame({'effect_sizes': effect_sizes,
#                               'pvals': pvals,
#                               'sample_sizes_wilcoxon': sample_sizes_wilcoxon,
#                               'sample_sizes_cohen': sample_sizes_cohen,
#                               'project': project_list,
#                               'colors': colors,
#                               'lead_descriptions': lead_descriptions,
#                               'annotations': annotations})

#         fig, axes = plt.subplots(1, 1, figsize=(20, 5))
#         sns.barplot(ax=axes, x='lead_descriptions', y='effect_sizes',
#                     data=stats,
#                     hue='project', palette=colors)
#         for patch in axes.patches:
#             width = patch.get_width()
#             # make bar for combined 'all' projects wider
#             if patch._facecolor == (0, 0, 0, 1):
#                 patch.set_width(width*2)
#         for idx, proj in enumerate(stats.project.unique()):
#             if proj != 'all':
#                 continue
#             labels = stats[(stats.project == proj)].annotations.to_list()
#             axes.bar_label(axes.containers[idx], labels=labels, fontsize=10,
#                            label_type='center', color='white')
#         sns.move_legend(axes, "upper left", bbox_to_anchor=(1, 1))
#         axes.set_ylabel('Cohen\'s d')
#         axes.set_xlabel('')
#         plt.suptitle(f'DBS directionality robustness {band}')
#         plt.tight_layout()
#         _save_fig(fig, save_path, cfg.FIG_RESULTS)


# def dbslead_manufacturer_barplot(df):
#     """Test robustness of effect sizes with regard to DBS leads.

#     Tests:
#     - Directional vs non-directional leads all manufacturers
#     - BS vs MT vs SJ for comparable leads
#     - any other important options?"""

#     save_dir = join('band_powers', 'robustness')
#     df = df[df.cond.isin(['on', 'off']) & (df.psd_kind == 'standard')
#             & df.ch_choice]
#     hue = 'cond'
#     dbs_manufacturers = df.DBS_manufacturer.dropna().unique()
#     bands = ['beta_low', 'HFO']
#     for band in bands:
#         save_path = join(save_dir, f'band_powers_robustness_leadmanus_{band}')
#         manufacturers = []
#         project_list = []
#         effect_sizes = []
#         colors = []
#         pvals = []
#         sample_sizes_wilcoxon = []
#         sample_sizes_cohen = []
#         annotations = []

#         projects = df.project.unique()
#         projects = [proj for proj in cfg.PROJECT_ORDER if proj in projects]
#         proj_slices = [df.project == col_val for col_val in projects]
#         projects.append('all')
#         proj_slices.append(slice(None))
#         for manu in dbs_manufacturers:
#             y = band + '_abs_max_log'
#             if band == 'HFO':
#                 y += "_nonoise"
#             for idx, proj in enumerate(projects):
#                 df_sub = df[proj_slices[idx]]
#                 df_sub = df_sub[df_sub.DBS_manufacturer == manu]
#                 effect_size, n_d = _cohen_stats(df_sub, y, hue)
#                 pval, n_wil = _wilcoxon_stats(df_sub, y, hue)
#                 effect_str = f"d={effect_size:.2f}\n(n={n_d})"
#                 pval = f"p={pval:1.0e}" if abs(pval) < 1e-2 else f"p={pval:.2f}"
#                 wilcoxon_str = f"\n{pval}\n(n={n_wil})"
#                 annotation = effect_str + wilcoxon_str

#                 effect_sizes.append(effect_size)
#                 sample_sizes_cohen.append(n_d)
#                 pvals.append(pval)
#                 sample_sizes_wilcoxon.append(n_wil)
#                 project_list.append(cfg.PROJECT_DICT[proj])
#                 colors.append(cfg.COLOR_DIC[proj])
#                 manufacturers.append(manu)
#                 annotations.append(annotation)

#         stats = pd.DataFrame({'effect_sizes': effect_sizes,
#                               'pvals': pvals,
#                               'sample_sizes_wilcoxon': sample_sizes_wilcoxon,
#                               'sample_sizes_cohen': sample_sizes_cohen,
#                               'project': project_list,
#                               'colors': colors,
#                               'dbs_manufacturers': manufacturers,
#                               'annotations': annotations})

#         fig, axes = plt.subplots(1, 1, figsize=(20, 5))
#         sns.barplot(ax=axes, x='dbs_manufacturers', y='effect_sizes',
#                     data=stats, hue='project', palette=colors)
#         for patch in axes.patches:
#             width = patch.get_width()
#             # make bar for combined 'all' projects wider
#             if patch._facecolor == (0, 0, 0, 1):
#                 patch.set_width(width*2)
#         for idx, proj in enumerate(stats.project.unique()):
#             if proj != 'all':
#                 continue
#             labels = stats[(stats.project == proj)].annotations.to_list()
#             axes.bar_label(axes.containers[idx], labels=labels,
#                         fontsize=10,
#                         label_type='center', color='white')
#         sns.move_legend(axes, "upper left", bbox_to_anchor=(1, 1))
#         axes.set_ylabel('Cohen\'s d')
#         axes.set_xlabel('')
#         plt.suptitle(f'DBS manufacturer robustness {band}')
#         plt.tight_layout()
#         _save_fig(fig, save_path, cfg.FIG_RESULTS)


# def dbslead_model_barplot(df):
#     """Test robustness of effect sizes with regard to DBS leads."""

#     save_dir = join('band_powers', 'robustness')
#     df = df[df.cond.isin(['on', 'off']) & (df.psd_kind == 'standard')
#             & df.ch_choice]
#     hue = 'cond'
#     dbs_models = df.DBS_model.dropna().unique()
#     bands = ['beta_low', 'HFO']
#     for band in bands:
#         save_path = join(save_dir, f'band_powers_robustness_leadmodels_{band}')
#         models = []
#         project_list = []
#         effect_sizes = []
#         colors = []
#         pvals = []
#         sample_sizes_wilcoxon = []
#         sample_sizes_cohen = []
#         annotations = []

#         projects = df.project.unique()
#         projects = [proj for proj in cfg.PROJECT_ORDER if proj in projects]
#         proj_slices = [df.project == col_val for col_val in projects]
#         projects.append('all')
#         proj_slices.append(slice(None))
#         for model in dbs_models:
#             y = band + '_abs_max_log'
#             if band == 'HFO':
#                 y += "_nonoise"
#             for idx, proj in enumerate(projects):
#                 df_sub = df[proj_slices[idx]]
#                 df_sub = df_sub[df_sub.DBS_model == model]
#                 effect_size, n_d = _cohen_stats(df_sub, y, hue)
#                 pval, n_wil = _wilcoxon_stats(df_sub, y, hue)
#                 effect_str = f"d={effect_size:.2f}\n(n={n_d})"
#                 pval = f"p={pval:1.0e}" if abs(pval) < 1e-2 else f"p={pval:.2f}"
#                 wilcoxon_str = f"\n{pval}\n(n={n_wil})"
#                 annotation = effect_str + wilcoxon_str

#                 effect_sizes.append(effect_size)
#                 sample_sizes_cohen.append(n_d)
#                 pvals.append(pval)
#                 sample_sizes_wilcoxon.append(n_wil)
#                 project_list.append(cfg.PROJECT_DICT[proj])
#                 colors.append(cfg.COLOR_DIC[proj])
#                 models.append(model)
#                 annotations.append(annotation)

#         stats = pd.DataFrame({'effect_sizes': effect_sizes,
#                               'pvals': pvals,
#                               'sample_sizes_wilcoxon': sample_sizes_wilcoxon,
#                               'sample_sizes_cohen': sample_sizes_cohen,
#                               'project': project_list,
#                               'colors': colors,
#                               'models': models,
#                               'annotations': annotations})

#         fig, axes = plt.subplots(1, 1, figsize=(20, 5))
#         sns.barplot(ax=axes, x='models', y='effect_sizes',
#                     data=stats, hue='project', palette=colors)
#         for patch in axes.patches:
#             width = patch.get_width()
#             # make bar for combined 'all' projects wider
#             if patch._facecolor == (0, 0, 0, 1):
#                 patch.set_width(width*2)
#         for idx, proj in enumerate(stats.project.unique()):
#             if proj != 'all':
#                 continue
#             labels = stats[(stats.project == proj)].annotations.to_list()
#             axes.bar_label(axes.containers[idx], labels=labels, fontsize=10,
#                            label_type='edge', color='k')
#         sns.move_legend(axes, "upper left", bbox_to_anchor=(1, 1))
#         axes.set_ylabel('Cohen\'s d')
#         axes.set_xlabel('')
#         plt.suptitle(f'DBS model robustness {band}')
#         plt.tight_layout()
#         _save_fig(fig, save_path, cfg.FIG_RESULTS)


def _patient_symptoms(df, save=True, conds=['off', 'on', 'offon_abs'],
                      save_dir=None, show_yticks=False):
    """Plot histogram of bradyrigid and tremor for all datasets."""
    df = df.drop_duplicates(subset=['project', 'subject', 'cond',
                                    'ch_hemisphere'])
    df = df[df.project != 'all']

    hue_order = [proj for proj in cfg.PROJECT_NAMES
                 if proj in df.project_nme.unique()]

    # indicate that tremor score missing for litvak and tan
    n_rows = len(conds)
    fig, axes = plt.subplots(n_rows, 3, figsize=(2.25, .75*n_rows),
                             sharex='col')
    axes = _axes2d(axes, n_rows, 3)
    kwargs = dict(hue='project_nme', palette=cfg.COLOR_DIC, common_norm=False,
                  hue_order=hue_order, legend=False,
                  common_grid=True, bw_method=0.5, cut=0.1)
    symptoms = ['UPDRS_bradyrigid_contra', 'UPDRS_tremor_contra', 'UPDRS_III']
    for j, cond in enumerate(conds):
        for i, symptom in enumerate(symptoms):
            no_tremor = ['Tan', 'Litvak'] if 'tremor' in symptom else []
            mask = (df.cond == cond) & ~df.project.isin(no_tremor)
            ax = axes[j, i]
            sns.kdeplot(ax=ax, data=df[mask], x=symptom, **kwargs)
    [ax.set_xlabel(None) for ax in axes.flatten()]
    [ax.set_ylabel(None) for ax in axes.flatten()]
    if show_yticks:
        [ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,
         decimals=0)) for ax in axes.flatten()]
    else:
        [ax.set_yticks([]) for ax in axes.flatten()]
        [ax.set_ylabel(f'Density {cfg.COND_DICT[conds[i]]}')
         for i, ax in enumerate(axes[:, 0])]
    axes[-1, 0].set_xlabel('Bradykinesia-rigidity')
    axes[-1, 1].set_xlabel('Tremor')
    axes[-1, 2].set_xlabel('Total UPDRS-III')
    plt.subplots_adjust(hspace=-0.3)  # reduce spacing between rows
    plt.tight_layout()
    if save:
        if save_dir is None:
            save_dir = join(cfg.FIG_RESULTS, 'patients', 'UPDRS')
        cond_str = '_'.join(conds)
        _save_fig(fig, f'patients_UPDRS_{cond_str}', save_dir, close=False,
                  bbox_inches=None)
    else:
        plt.show()


def _patient_symptoms_flat(df, conds=['off', 'on', 'offon_abs'],
                           fig_dir=None, show_yticks=False, prefix=''):
    """Plot histogram of bradyrigid and tremor for all datasets."""
    df = df.drop_duplicates(subset=['project', 'subject', 'cond',
                                    'ch_hemisphere'])

    hue_order = [proj for proj in cfg.PROJECT_NAMES
                 if proj in df.project_nme.unique()]

    # indicate that tremor score missing for litvak and tan
    symptoms = ['UPDRS_bradyrigid_contra', 'UPDRS_tremor_contra', 'UPDRS_III']
    symptom_dic = {'UPDRS_bradyrigid_contra': 'Bradykinesia-rigidity',
                   'UPDRS_tremor_contra': 'Tremor',
                   'UPDRS_III': 'Total UPDRS-III'}
    n_rows = 1
    n_cols = len(conds) * len(symptoms)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2, 1))
    axes = _axes2d(axes, n_rows, n_cols)
    kwargs = dict(hue='project_nme', palette=cfg.COLOR_DIC, common_norm=False,
                  hue_order=hue_order, legend=False,
                  common_grid=True, bw_method=0.5, cut=0.1)
    conds_symptoms = list(product(conds, symptoms))
    for i, (cond, symptom) in enumerate(conds_symptoms):
        no_tremor = ['Tan', 'Litvak'] if 'tremor' in symptom else []
        mask = (df.cond == cond) & ~df.project.isin(no_tremor)
        ax = axes[0, i]
        sns.kdeplot(ax=ax, data=df[mask], x=symptom, **kwargs)
        ax.set_xlabel(symptom_dic[symptom])
        ax.set_ylabel(None)
    if show_yticks:
        [ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,
         decimals=0)) for ax in axes.flatten()]
    else:
        [ax.set_yticks([]) for ax in axes.flatten()]
        [ax.set_ylabel(f'Density {cfg.COND_DICT[conds[i]]}')
         for i, ax in enumerate(axes[:, 0])]
    axes[0, 0].set_ylabel('Density')

    # Add extra space between groups (1-3, 4-6, 7-9)
    for idx in [2, 5]:  # Indices after each group
        fig.subplots_adjust(left=.1)  # Adjust left side by increments

    plt.tight_layout()
    cond_str = '_'.join(conds)
    _save_fig(fig, f'{prefix}patients_UPDRS_{cond_str}',
              join(cfg.FIG_PAPER, fig_dir), close=False,
              transparent=True, bbox_inches=None)


def _patient_demographics(df, save=True, save_dir=None, show_yticks=False):
    """Plot patient info for all datasets."""
    df = df.drop_duplicates(subset=['project', 'subject'])

    density_cols = ['patient_age', 'patient_disease_duration']
    # columns without variance cannot be plotted with kdeplot
    hist_cols = ['patient_sex']
    n_cols = len(density_cols) + len(hist_cols)
    hue_order = [proj for proj in cfg.PROJECT_NAMES
                 if proj in df.project_nme.unique()]

    fig, axes = plt.subplots(1, n_cols, figsize=(2, .75))
    kwargs = dict(hue='project_nme', palette=cfg.COLOR_DIC,
                  legend=False,
                  common_norm=False,  # True overemphasizes 'all'
                  common_grid=True, bw_method=0.5, cut=0.1)
    for j, info in enumerate(density_cols):
        ax = axes[j]
        sns.kdeplot(ax=ax, data=df, x=info, **kwargs)
        if show_yticks:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,
                                                                decimals=0))
            if j > 0:
                ax.set_yticklabels([])
    # don't plot unknown sex because not informative
    df = df[df.patient_sex.isin(['male', 'female'])]
    df['patient_sex'] = df['patient_sex'].map({'male': 'M', 'female': 'F'})
    for j, info in enumerate(hist_cols):
        ax = axes[j + len(density_cols)]
        sns.histplot(ax=ax, data=df, x=info, hue='project_nme',
                     palette=cfg.COLOR_DIC,
                     multiple='dodge', legend=False,
                     shrink=0.8,
                     hue_order=hue_order, stat='percent',
                     common_norm=False)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylabel(None)
    axes[0].set_xlabel('Age [yrs]')
    axes[1].set_xlabel('PD duration [yrs]')
    axes[2].set_xlabel('Sex')
    if not show_yticks:
        axes[0].set_ylabel('Density')
        axes[0].set_yticks([])
        axes[1].set_yticks([])
    axes[1].set_ylabel(None)
    plt.tight_layout()
    if save:
        if save_dir is None:
            save_dir = join(cfg.FIG_RESULTS, 'patients', 'demographics')
        _save_fig(fig, 'patients_demographics', save_dir, close=False)
    else:
        plt.show()


def _patient_recording_and_leads(df, save=True, save_dir=None):
    df = df[~df.project.isin(['all'])]
    df = df.drop_duplicates(subset=['subject'])

    df.loc[df.DBS_directional, 'ch_dir'] = 'Yes'
    df.loc[~df.DBS_directional, 'ch_dir'] = 'No'

    hist_cols = ['patient_days_after_implantation', 'ch_dir']
    n_cols = len(hist_cols)
    hue_order = [proj for proj in cfg.PROJECT_NAMES
                 if proj in df.project_nme.unique()]

    fig, axes = plt.subplots(1, n_cols, figsize=(1.5, .75), sharey=False)
    for j, info in enumerate(hist_cols):
        ax = axes[j]
        sns.histplot(ax=ax, data=df, x=info, hue='project_nme',
                     palette=cfg.COLOR_DIC, multiple='stack', legend=False,
                     shrink=0.8, hue_order=hue_order, stat='percent',
                     common_norm=True, discrete=True)
        if info == 'patient_days_after_implantation':
            xticks = range(0, 8)
            ax.set_xticks(xticks)
    axes[0].set_xlabel('Day of recording')
    axes[1].set_xlabel('DBS directional')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    axes[0].set_ylabel(None)
    axes[1].set_ylabel(None)
    plt.tight_layout()
    if save:
        if save_dir is None:
            save_dir = join(cfg.FIG_RESULTS, 'patients', 'DBSleads')
        _save_fig(fig, 'patients_recordings', save_dir, close=False)
    else:
        plt.show()


def _patient_sample_size(df, save=True, save_dir=None):
    df = df.copy()
    hue_order = [proj for proj in cfg.PROJECT_NAMES
                 if proj in df.project_nme.unique()]

    # cond order
    df['cond'] = pd.Categorical(df['cond'], ['off', 'on', 'offon_abs'])
    rename = {'off': 'off', 'on': 'on', 'offon_abs': 'off+on'}
    df['cond'] = df['cond'].map(rename)

    duplicates = ['project', 'subject', 'cond']
    df_hemi = df.drop_duplicates(subset=duplicates + ['ch_hemisphere'])
    df_sub = df.drop_duplicates(subset=duplicates)

    kwargs = dict(hue='project_nme', palette=cfg.COLOR_DIC,
                  multiple='dodge', x='cond', hue_order=hue_order,
                  stat='count', legend=False, shrink=0.8)

    fig, ax = plt.subplots(1, 1, figsize=(1, .75))
    sns.histplot(ax=ax, data=df_sub, **kwargs)
    ax.set_ylabel('# Patients')
    ax.set_xlabel(None)
    plt.tight_layout()
    if save:
        if save_dir is None:
            save_dir = join(cfg.FIG_RESULTS, 'patients', 'DBSleads')
        _save_fig(fig, 'patients_sample_size', save_dir, close=False)
    else:
        plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(1, .75))
    sns.histplot(ax=ax, data=df_hemi, **kwargs)
    ax.set_ylabel('# Hemispheres')
    ax.set_xlabel(None)
    plt.tight_layout()
    if save:
        if save_dir is None:
            save_dir = join(cfg.FIG_RESULTS, 'patients', 'DBSleads')
        _save_fig(fig, 'hemis_sample_size', save_dir, close=False)
    else:
        plt.show()


def _wilcoxon_stats(df_sub, y, hue, ch_selection=None, mask=None):
    from statannotations.stats.StatTest import wilcoxon
    if not len(df_sub) or df_sub[hue].nunique() < 2:
        return np.nan, 0
    if ch_selection == 'Max. Beta':
        # otherwise equalize x and y will fail if chs different in ON and OFF
        rename_L = (mask & (df_sub.ch_hemisphere == 'L'), 'ch_nme')
        rename_R = (mask & (df_sub.ch_hemisphere == 'R'), 'ch_nme')
        df_sub.loc[rename_L] = 'LFP_L_BetaMax'
        df_sub.loc[rename_R] = 'LFP_R_BetaMax'
    df_sub, n_wil = equalize_x_and_y(df_sub, hue, y)
    x1 = df_sub[(df_sub.cond == 'off')][y]
    x2 = df_sub[(df_sub.cond == 'on')][y]
    pval = wilcoxon(x1, x2).pvalue
    return pval, n_wil


def _cohen_stats(df_sub, y, hue):
    if not len(df_sub) or df_sub[hue].nunique() < 2:
        return np.nan, 0
    group = df_sub[[hue, y]].groupby(hue)
    x1, x2 = group[y].apply(lambda x: x.values)
    effect_size = cohen_d(x1, x2)
    n_d = (len(x1) + len(x2)) / 2
    return effect_size, n_d


def plot_psd_units(raw, title='Amplifier'):
    import scipy.signal as sig
    fmax = raw.info["lowpass"]
    freqs, psd = sig.welch(raw.get_data(), fs=raw.info["sfreq"],
                           nperseg=raw.info["sfreq"])
    mask = freqs <= fmax
    freqs = freqs[mask]
    psd = psd[:, mask]

    asd = (psd**.5) * 1e9  # convert V**2/Hz to nV/sqrt(Hz)

    global asd_amp
    global asd_ch_names
    global freqs_amp
    asd_amp = asd
    asd_ch_names = raw.ch_names
    freqs_amp = freqs

    fig, axes = plt.subplots(1, 4, figsize=(25, 10))
    for idx, ch_nme in enumerate(raw.ch_names):
        if ch_nme in raw.info['bads']:
            continue
        ch_splits = ch_nme.split('_')

        # monopolar
        if ch_splits[2] in ['1', '2', '3', '4']:
            axes[0].loglog(freqs, asd[idx], label=ch_nme + ' mono')
        # nondir-nondir
        elif ch_splits[2] in ['1-4']:
            axes[1].loglog(freqs, asd[idx], label=ch_nme + ' nondir')
        # dir-dir
        elif ch_splits[2] in ['2-3']:
            axes[2].loglog(freqs, asd[idx], label=ch_nme + ' dir-dir')
        # dir-nondir
        elif ch_splits[2] in ['1-2', '3-4']:
            axes[3].loglog(freqs, asd[idx], label=ch_nme + ' dir-nondir')

    for idx, ch_nme in enumerate(asd_ch_names):
        if ch_nme in raw.info['bads']:
            continue
        ch_splits = ch_nme.split('_')

        # monopolar
        if ch_splits[2] in ['1', '2', '3', '4']:
            axes[0].loglog(freqs_amp, asd_amp[idx], label=ch_nme + ' mono')
        # nondir-nondir
        elif ch_splits[2] in ['1-4']:
            axes[1].loglog(freqs_amp, asd_amp[idx], label=ch_nme + ' nondir')
        # dir-dir
        elif ch_splits[2] in ['2-3']:
            axes[2].loglog(freqs_amp, asd_amp[idx], label=ch_nme + ' dir-dir')
        # dir-nondir
        elif ch_splits[2] in ['1-2', '3-4']:
            axes[3].loglog(
                freqs_amp, asd_amp[idx], label=ch_nme + ' dir-nondir'
            )

    axes[0].set_title('monopolar')
    axes[1].set_title('bipolar non-directional')
    axes[2].set_title('bipolar directional')
    axes[3].set_title('bipolar mixed')

    yticks = [10, 20, 30, 40, 50, 100, 500, 1000, 10000, 20000]
    for ax in axes:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylim(yticks[0], yticks[-1])
        ax.legend()
    axes[0].legend(ncols=2)
    axes[0].set_ylabel(r'ASD [$nV/\sqrt{Hz}$]')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def convert_pvalue_to_asterisks(pvalue,
                                stack_vertically=False,
                                underline=False, print_ns=False):
    if pvalue <= 0.001:
        if stack_vertically:
            return "*\n*\n*"
        elif underline:
            return r"\underline{***}"
        else:
            return "***"
    elif pvalue <= 0.01:
        if stack_vertically:
            return "*\n*"
        elif underline:
            return r"\underline{**}"
        else:
            return "**"
    elif pvalue <= 0.05:
        if underline:
            return r"\underline{*}"
        else:
            return "*"
    if print_ns:
        return "ns"
    return ""


def _stat_anno(ax, df, x, y, groupby='subject', alternative='two-sided',
               y_line=None, fontsize=7):
    ymin, ymax = ax.get_ylim()
    yscale = np.abs(ymax - ymin)
    y_buffer = 0.05*yscale

    # Draw the line connecting the two bars
    x1, x2 = ax.get_xticks()
    if y_line is None:
        y_line = ymax + y_buffer
    ax.plot([x1, x1, x2, x2],
            [y_line, y_line + y_buffer/2, y_line + y_buffer/2, y_line],
            color='black')

    # Get the significance text based on the p-value
    vals = df.copy().sort_values([groupby, x]).groupby(groupby)[y]
    xy_diff = vals.diff().dropna()
    pvalue = wilcoxon(xy_diff, alternative=alternative)[1]
    # print(pvalue)
    text = convert_pvalue_to_asterisks(pvalue, print_ns=True)

    # Place the text above the line
    y_text = y_line + y_buffer/2  # Add a little more offset for the asterisks
    ax.text((x1 + x2) / 2, y_text, text, ha='center', va='bottom',
            fontsize=fontsize)

    if y_line is None:
        ax.set_ylim([ymin-y_buffer, ymax+4*y_buffer])
