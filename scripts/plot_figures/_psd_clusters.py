"""Helping functions."""
from itertools import product
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from pte_stats import cluster, timeseries

import scripts.config as cfg
from scripts.plot_figures.settings import *
from scripts.utils_plot import _save_fig, explode_df


def _remove_single_hemi_subs(df):
    """Remove subject with only one hemisphere."""
    subject_counts = df.subject.value_counts()
    valid_subjects = subject_counts[subject_counts == 2].index
    df = df[df.subject.isin(valid_subjects)]
    return df


def _extract_arrays(df1, df2, x, y, x_max, x_min):
    if 'fm' in x:
        # different projects have different fitting ranges
        cond_arr1 = []
        for _, row1 in df1.iterrows():
            if np.isnan(row1[y]).any():
                raise ValueError("NaNs should not be present")
            times = row1[x]
            x_mask = (times >= x_min) & (times <= x_max)
            times = times[x_mask]
            cond_arr1.append(row1[y][x_mask])
        # use a separate loop since df1 and df2 can have different lengths
        # for non-paired statistics
        cond_arr2 = []
        for _, row2 in df2.iterrows():
            if np.isnan(row2[y]).any():
                raise ValueError("NaNs should not be present")
            times = row2[x]
            x_mask = (times >= x_min) & (times <= x_max)
            times = times[x_mask]
            cond_arr2.append(row2[y][x_mask])

        x_arr = np.stack(cond_arr1).T
        y_arr = np.stack(cond_arr2).T
    else:
        times = df1[x].values[0]
        x_mask = (times >= x_min) & (times <= x_max)
        times = times[x_mask]

        x_arr = np.stack(df1[y].values)[:, x_mask].T
        y_arr = np.stack(df2[y].values)[:, x_mask].T
    return x_arr, y_arr, times


def _extract_arrays_old(df1, df2, x, y, plot_max, x_min, x_max):
    if 'fm' in x:
        # different projects have different fitting ranges
        cond_arr1 = []
        for _, row1 in df1.iterrows():
            if np.isnan(row1[y]).any():
                continue
            times = row1[x]
            # assert row2[x].all() == times.all()
            x_mask = (times >= x_min) & (times <= plot_max)
            times = times[x_mask]
            cond_arr1.append(row1[y][x_mask])
        # use a separate loop since df1 and df2 can have different lengths
        # for non-paired statistics
        cond_arr2 = []
        for _, row2 in df2.iterrows():
            if np.isnan(row2[y]).any():
                continue
            times = row2[x]
            # assert row2[x].all() == times.all()
            x_mask = (times >= x_min) & (times <= plot_max)
            times = times[x_mask]
            cond_arr2.append(row2[y][x_mask])

        x_arr = np.stack(cond_arr1).T
        y_arr = np.stack(cond_arr2).T
    else:
        times = df1[x].values[0]
        x_mask = (times >= x_min) & (times <= x_max)
        times = times[x_mask]

        x_arr = np.stack(df1[y].values)[:, x_mask].T
        y_arr = np.stack(df2[y].values)[:, x_mask].T
    return x_arr, y_arr, times


def _get_clusters(x1, x2, times, output_file, alpha_sig=0.05,
                  n_perm=N_PERM_CLUSTER, paired_x1x2=True):
    cluster_times = []
    two_tailed = True
    one_tailed_test = 'larger'
    min_cluster_size = 2
    if paired_x1x2:
        data_a = x1 - x2
        data_b = 0.0
    else:
        data_a = x1
        data_b = x2
    if not two_tailed and one_tailed_test == "smaller":
        data_a_stat = data_a * -1
    else:
        data_a_stat = data_a

    p_vals = timeseries.timeseries_pvals(
            x=data_a_stat, y=data_b, n_perm=n_perm, two_tailed=two_tailed)
    clusters, cluster_count = cluster.clusters_from_pvals(
            p_vals=p_vals, alpha=alpha_sig, correction_method='cluster_pvals',
            n_perm=n_perm, min_cluster_size=min_cluster_size)

    x_labels = times.round(2)
    for cluster_idx in range(1, cluster_count + 1):
        index = np.where(clusters == cluster_idx)[0]
        if index.size == 0:
            print("No clusters found.", file=output_file)
            continue
        lims = np.arange(index[0], index[-1] + 1)
        time_0 = x_labels[lims[0]]
        time_1 = x_labels[lims[-1]]
        print(f"Cluster found between {time_0} Hz and" f" {time_1} Hz.",
              file=output_file)
        cluster_times.append((time_0, time_1))
    cluster_borders = np.array(cluster_times)
    return clusters, cluster_count, cluster_borders


def _plot_clusters(ax, x1, x2, times, clusters, cluster_count,
                   alpha=0.05, color_cluster='k', alpha_plot=.2):

    if isinstance(x2, (int, float)):
        y_arr = np.ones((x1.shape[0], 1))
        y_arr[:, 0] = x2
    else:
        y_arr = x2
    x_arr = x1
    label = f"p ≤ {alpha}"
    for cluster_idx in range(1, cluster_count + 1):
        index = np.where(clusters == cluster_idx)[0]
        if index.size == 0:
            continue
        lims = np.arange(index[0], index[-1] + 1)
        y1 = x_arr.mean(axis=1)[lims]
        y2 = y_arr.mean(axis=1)[lims]
        ax.fill_between(x=times[lims], y1=y1, y2=y2, alpha=alpha_plot,
                        color=color_cluster,
                        label=label)


def _set_yscale(ax, cond, kind, yscale, info_title, yticks, yticklabels=None):
    if yscale is None:
        if cond == 'offon_abs':
            yscale = 'symlog'
        elif cond in ['off', 'on']:
            yscale = 'log'
    ax.set_yscale(yscale)
    color = cfg.COLOR_DIC['periodicAP'] if kind == 'periodicBOTH' else 'k'
    if 'log' in yscale and kind != 'periodic':
        formatter = lambda x, pos: f'{x:.1f}'.rstrip('0').rstrip('.')
        ax.yaxis.set_major_formatter(FuncFormatter(formatter))
        ax.tick_params(which='major', axis='y', labelcolor=color)
    if info_title == False:
        if yticks is None:
            yticks = ax.get_yticks()
    else:
        yticklabels = yticks
    if yticklabels is None:
        yticklabels = yticks
    elif yticklabels == False:
        yticklabels = ['' for _ in yticks]
    elif isinstance(yticklabels, list):
        pass
    if yticks is not None:
        ax.set_yticks(yticks, labels=yticklabels)


def _annotate_stats(ax, all_clusters, color=None, labels=None,
                    total_stats=False, yscale='linear', height=None):
    """Plot cluster significance as horizontal lines above the x-axis."""
    if not all_clusters:
        return
    if color is None:
        colors = ['k'] * len(all_clusters)
    elif isinstance(color, str):
        colors = [color] * len(all_clusters)
    elif isinstance(color, tuple):
        # RGB tuple
        colors = [color] * len(all_clusters)
    else:
        msg = f"color must be str or list, got {color} which is {type(color)}"
        assert isinstance(color, list), msg
        colors = color
    if labels is None:
        labels = [None] * len(all_clusters)
    # all_clusters must be list of lists of tuples.
    # Tuples: cluster borders, Lists: cluster_borders per data input
    if isinstance(all_clusters[0], list):
        pass
    elif isinstance(all_clusters[0], tuple):
        all_clusters = [all_clusters]

    if yscale == 'log':
        if height is not None:
            height_axes = height
        else:
            height_axes = ax.get_ylim()[0]
    elif yscale == 'linear':
        height_axes = 0.015
    shift = 0.02
    height_data = ax.transAxes.transform((0, height_axes))[1]
    height_data = ax.transData.inverted().transform((0, height_data))[1]
    original_ylims = ax.get_ylim()
    # reverse since plotted down to up
    all_clusters = all_clusters[::-1]
    labels = labels[::-1]
    colors = colors[::-1]
    for i, cluster_borders in enumerate(all_clusters):
        label = labels[i]
        if label == 'Total' and not total_stats:
            continue
        if len(cluster_borders):
            for (lim1, lim2) in cluster_borders:
                ax.plot([lim1, lim2], [height_data, height_data],
                        color=colors[i], lw=1, label=labels[i])
            # Adjust the height for the next annotation
            height_axes += shift
            height_data = ax.transAxes.transform((0, height_axes))[1]
            height_data = ax.transData.inverted().transform((0, height_data))[1]
    # shrink yaxis to prevent adjustment to annotated stats
    ax.set_ylim(*original_ylims)


def plot_psd_by_severity_conds(dataframes, kind, conds=['off', 'on'],
                               within_comparison=False, n_perm=N_PERM_CLUSTER,
                               xscale='linear', hemispheres=None,
                               yscale='linear', ylim=None,
                               figsize=(1.5, 1.5),
                               lateralized_updrs=False, color_by_kind=True,
                               legend=True, xlabel=True, ylabel=True,
                               yticks=None, yticklabels=None,
                               xticks=None, xticklabels=None,
                               fig_dir='Figure5', stat_height=None, prefix='',
                               xmin=2, xmax=60, info_title=None,
                               leg_kws=dict(), output_file=None):
    # Checks
    if 'off' in conds and 'on' in conds:
        msg = f"conds must be {cfg.COND_ORDER}"
        assert conds == cfg.COND_ORDER[:len(conds)], msg
    assert len(conds) <= 2, f"Only 2 conditions allowed, got {len(conds)}"
    if 'offon' in conds and within_comparison:
        msg = ("Off-On not implemented within subject. To do, run "
               "_updrs_ipsi_contra() in _04_organize_dataframe.py after "
               "running _offon_recording()")
        raise NotImplementedError(msg)
    if within_comparison:
        assert lateralized_updrs, "Within requires lateralized UPDRS."
        assert hemispheres is None

    # Extract plot settings and variables
    # always same settings except for periodicBOTH
    plot_lines = [True]
    plot_clusters = [False]
    color_cluster = 'k'
    if kind in ['normalized', 'absolute', 'normalizedInce']:
        palette = [[cfg.COLOR_DIC[kind]], [cfg.COLOR_DIC[kind + '2']]]
        legend_palette = palette
        x = 'psd_freqs'
        y_vals = ['psd']
        labels = [cfg.KIND_DICT[kind]]
        if kind == 'normalized':
            df = dataframes['df_norm']
            if not color_by_kind:
                # color by project
                palette = [[cfg.COLOR_DIC['all']], [cfg.COLOR_DIC['all']]]
                legend_palette = palette
                color_cluster = 'y'
            ylabel = 'Spectrum [%]' if ylabel else None
        elif kind == 'normalizedInce':
            df = dataframes['df_plateau']
            ylabel = 'Spectrum [%]' if ylabel else None
        elif kind  == 'absolute':
            df = dataframes['df_abs']
            ylabel = 'Spectrum 'r'[$\mu V^2/Hz$]' if ylabel else None
            if ylim is None:
                ylim = (0.015, 5)
    elif kind.startswith('periodic'):
        df = dataframes['df_per']
        x = 'fm_freqs'
        y_kind_dict = {'fm_psd_peak_fit': 'periodic',
                        'fm_psd_ap_fit': 'periodicAP',
                        'fm_fooofed_spectrum': 'periodicFULL'}
        if ylim is None:
            ylim = (-0.035, 1)
        palette = [[cfg.COLOR_DIC[kind]], [cfg.COLOR_DIC[kind + '2']]]
        legend_palette = palette
        if kind == 'periodic':
            y_vals = ['fm_psd_peak_fit']
        elif kind == 'periodicAP':
            y_vals = ['fm_psd_ap_fit']
            if ylim is None:
                ylim = (0.015, 5)
        elif kind == 'periodicFULL':
            y_vals = ['fm_fooofed_spectrum']
        labels = [cfg.KIND_DICT[y_kind_dict[y]] for y in y_vals]
        ylabel = 'Spectrum 'r'[$\mu V^2/Hz$]' if ylabel else None
    else:
        raise ValueError(f"Unknown kind {kind}")

    df = df[(df.project == 'all')]
    labels_conds = [labels, labels]  # duplicate to avoid altering
    fig, axes = plt.subplots(1, len(conds), figsize=figsize, sharey=True)
    for cond in conds:
        ic = 0 if cond == 'off' else 1
        if isinstance(axes, np.ndarray):
            ax = axes[0]
        else:
            ax = axes
        colors = palette[ic]
        legend_colors = legend_palette[ic]
        labels = labels_conds[ic]
        df_cond = df[(df.cond == cond)]
        # for OFF-ON, use log values to make comparable to log peak power
        # OFF-ON
        if cond == 'offon_abs':
            y_vals = [y + '_log' for y in y_vals]
            ylim = None
        else:
            # for OFF and ON values, use linear units. Important: Plot and
            # statistics are different when performed on psd_log vs psd. The
            # stats are based on the mean of the psd. mean(psd_a) vs
            # mean(psd_b) on a log scale is different from mean(log(psd_a)) vs
            # mean(log(psd_b)). Linear values on log scale appear more
            # reasonable since any errors based on non-linearities can be
            # excluded.
            pass

        # drop nan arrays (df.dropna(y) does not work for arrays)
        for y in y_vals:
            mask = pd.Series(df_cond[y].apply(lambda x: np.all(np.isnan(x))))
            df_cond = df_cond[~mask]

        if lateralized_updrs:
            updrs = 'UPDRS_bradyrigid_contra'
        else:
            updrs = 'UPDRS_III'
        if within_comparison:
            df_cond = df_cond[df_cond.fm_exponent.notna()]
            # Stat settings
            paired_x1x2 = True
            hue = f"patient_symptom_dominant_side_BR_{cond}"
            # remove subjects without asymmetry
            # rename rows based on dictionary
            rename = {'severe side': 'More affected',
                      'mild side': 'Less affected'}
            hue_order = ['More affected', 'Less affected']
            if cond == 'on':
                # only include consistent asymmetry for ON subjects to
                # exclude possible LDOPA side effects
                df_cond = df_cond[df_cond.dominant_side_consistent]
        else:
            # Stat settings
            paired_x1x2 = False
            # rename to ease interpretation
            if cond in ['off', 'on']:
                sampling = 'STNs' if lateralized_updrs else 'patients'
                rename = {'mild_half': f'Mild {sampling}',
                          'severe_half': f'Severe {sampling}'}
                hue_order = [f'Severe {sampling}', f'Mild {sampling}']
            elif cond.startswith('offon'):
                rename = {'mild_half': 'Weak responders',
                          'severe_half': 'Strong responders'}
                hue_order = ['Weak responders', 'Strong responders']
            hue = f'{updrs}_severity_median'
        df_cond[hue] = df_cond[hue].astype('category')
        df_cond[hue] = df_cond[hue].cat.rename_categories(rename)
        df_cond[hue] = df_cond[hue].cat.remove_unused_categories()
        df_cond = df_cond[df_cond[hue].isin(hue_order)]

        if within_comparison:
            # Plot settings
            cluster_str = 'within'
            # does nothing if asymmetric subjects selected:
            df_cond = _remove_single_hemi_subs(df_cond)
            hemisphere_str = ''
        else:
            # Plot settings
            if hemispheres is None:
                if not lateralized_updrs:
                    # average hemispheres
                    group = df_cond.groupby(['subject'])
                    df_cond[x] = group[x].transform("mean")
                    for y in y_vals:
                        df_cond[y] = group[y].transform("mean")
                    df_cond = df_cond.drop_duplicates(subset=["subject"])
                    cluster_str = 'across_mean'
                else:
                    cluster_str = 'across'
                hemisphere_str = ''
            elif hemispheres in ['severe side', 'mild side']:
                # only keep selected hemispheres
                df_cond = df_cond[df_cond.patient_symptom_dominant_side_BR
                                  == hemispheres]
                cluster_str = 'across_' + hemispheres.replace(' ', '_')
                hemisphere_str = '_' + hemispheres.replace(' ', '_')

        # Extract cluster varible
        cluster_conds = df_cond[hue].unique()
        assert len(cluster_conds) == 2

        df1 = df_cond[df_cond[hue] == cluster_conds[0]]
        df2 = df_cond[df_cond[hue] == cluster_conds[1]]

        if within_comparison:
            assert (df1.subject.to_numpy() == df2.subject.to_numpy()).all()

        # Plotting
        all_clusters = []
        periodic_stats = []  # here it gets complicated. I need to transfer the
        # periodic cluster stats to full plot in order have both correct stats
        # and a correct visualization.
        for i, y in enumerate(y_vals):
            x_array, y_array, times = _extract_arrays(df1, df2, x, y, xmax,
                                                      xmin)

            if plot_lines[i]:
                # do not plot fitted peaks without aperiodic power
                df_exploded = explode_df(df_cond, freqs=x, psd=y,
                                         keep_cols=[hue, 'sub_hemi'],
                                         fmax=xmax, fmin=xmin)

                n_x = x_array.shape[1]
                n_y = y_array.shape[1]
                assert df_exploded[df_exploded[hue]
                                   == cluster_conds[0]
                                   ].sub_hemi.nunique() == n_x
                assert df_exploded[df_exploded[hue]
                                   == cluster_conds[1]
                                   ].sub_hemi.nunique() == n_y

                sns.lineplot(data=df_exploded, x=x, y=y, hue=hue, ax=ax,
                             errorbar=CI_SPECT, hue_order=hue_order,
                             palette=[colors[i], colors[i]],
                             style=hue, style_order=hue_order)

            print(f'\n{kind} {cond} (<{xmax} Hz):\n', file=output_file)
            clusters, cluster_count, cluster_borders = _get_clusters(
                x_array, y_array, times, output_file, n_perm=n_perm,
                paired_x1x2=paired_x1x2
                )

            all_clusters.append(cluster_borders)
            if y.startswith('fm_'):
                periodic_stats.append((clusters, cluster_count))
            if plot_clusters[i]:
                if y.startswith('fm_fooofed_spectrum'):
                    clusters, cluster_count = periodic_stats[0]
                _plot_clusters(ax, x_array, y_array, times, clusters,
                               cluster_count, color_cluster=color_cluster,
                               alpha_plot=0.4)

        if cond == 'offon_abs':
            if lateralized_updrs:
                sample = cfg.SAMPLE_STN
            else:
                sample = cfg.SAMPLE_PAT
        else:
            sample = 'n'
        if within_comparison:
            assert n_x == n_y
            sample_size_str1 = f'({sample}={n_x})'
            sample_size_str2 = sample_size_str1
        else:
            sample_size_str1 = f'({sample}={n_x})'
            sample_size_str2 = f'({sample}={n_y})'
        cond_str = cfg.COND_DICT[cond]
        if info_title:
            kind_str = cfg.KIND_DICT[kind]
            updrs_str = cfg.PLOT_LABELS[updrs]
            title = f'{kind_str}, {cond_str}, {updrs_str}'
            ax.set_title(title)
            info_str = '_LongTitle'
        elif info_title is None:
            ax.set_title(cond_str)
            info_str = '_Title'
        elif info_title is False:
            info_str = ''
        ax.set_ylim(ylim)
        _annotate_stats(ax, all_clusters, legend_colors, labels, yscale=yscale,
                        height=stat_height)
        if legend:
            if kind == 'periodicBOTH':
                handles = [Line2D([0], [0], color=color, ls='-')
                            for color in legend_colors]
                total_stats = False  # do not show cluster stats for full model
                if not total_stats:
                    labels = labels[:-1]
                    handles = handles[:-1]
                title = None
            else:
                # Plot neutral legend without kind colors
                handles = [Line2D([0], [0], color='k', ls=ls) for ls in ['-', '--']]
                labels = [f'{hue_order[0]} {sample_size_str1}',
                          f'{hue_order[1]} {sample_size_str2}']
                # Plot legend in kind colors
                if within_comparison:
                    title = 'Hemisphere'
                else:
                    title = None
            ax.legend(handles, labels, handlelength=1, title=title, **leg_kws)
            legend = False  # only on first axis
        else:
            ax.get_legend().remove()
    if xlabel:
        ax.set_xlabel("Frequency [Hz]")
    else:
        ax.set_xlabel(None)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xticks is not None:
        ax.set_xticks(ticks=xticks, labels=xticklabels)
    if yticks is not None:
        ax.set_yticks(ticks=yticks, labels=yticklabels)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ylim)
    if ax.get_yscale() == 'log':
        ax.yaxis.set_tick_params(which='major', pad=0)
    if ax.get_xscale() == 'log':
        ax.xaxis.set_tick_params(which='major', pad=0)
    plt.minorticks_off()
    cond_str = '_'.join(conds)
    save_dir = join(cfg.FIG_PAPER, fig_dir)
    fname = (f'{prefix}psd_clusters_{cluster_str}_{kind}_{hue}_'
             f'{xmax}Hz_{cond_str}_'
             f'{xscale}_{yscale}{hemisphere_str}'
             f'{info_str}')
    plt.tight_layout()
    _save_fig(fig, fname, save_dir, bbox_inches=None,
              facecolor=(1, 1, 1, 0))


def plot_psd_by_severity_kinds(dataframes, kinds, conds=['off', 'on'],
                               within_comparison=False, n_perm=N_PERM_CLUSTER,
                               xscale='linear',
                               ylim_norm=None,
                               ylim_per=(-0.035, 1), ylim_ap=(0.015, 5),
                               ylim_abs=(0.015, 5),
                               figsize=(1.5, 1.5),
                               lateralized_updrs=False, color_by_kind=True,
                               legend=True, xlabel=True, ylabel=True,
                               fig_dir='Figure5', yticks=None,
                               stat_height=None,
                               prefix='',
                               xmin=2, xmax=60, info_title=None,
                               leg_kws=dict(),
                               output_file=None):
    # Checks
    if 'off' in conds and 'on' in conds:
        msg = f"conds must be {cfg.COND_ORDER}"
        assert conds == cfg.COND_ORDER[:len(conds)], msg
    assert len(conds) <= 2, f"Only 2 conditions allowed, got {len(conds)}"
    if 'offon' in conds and within_comparison:
        msg = ("Off-On not implemented within subject. To do, run "
               "_updrs_ipsi_contra() in _04_organize_dataframe.py after "
               "running _offon_recording()")
        raise NotImplementedError(msg)
    if within_comparison:
        assert lateralized_updrs, "Within requires lateralized UPDRS."

    # Extract plot settings and variables
    # always same settings except for periodicBOTH
    plot_lines = [True]
    plot_clusters = [False]
    color_cluster = 'k'
    periodic_shift = 0

    kinds_conds = list(product(kinds, conds))
    n_cols = len(kinds_conds)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=False)

    for idx, (kind, cond) in enumerate(kinds_conds):
        ax = axes[idx]
        yscale = 'log'  # always log except pure periodic and relative
        if kind in ['normalized', 'absolute', 'normalizedInce']:
            palette = [[cfg.COLOR_DIC[kind]], [cfg.COLOR_DIC[kind + '2']]]
            legend_palette = palette
            x = 'psd_freqs'
            y_vals = ['psd']
            labels = [cfg.KIND_DICT[kind]]
            if kind == 'normalized':
                df = dataframes['df_norm']
                if not color_by_kind:
                    # color by project
                    palette = [[cfg.COLOR_DIC['all']], [cfg.COLOR_DIC['all2']]]
                    legend_palette = palette
                    color_cluster = 'y'
                ylabel = 'Spectrum [%]' if ylabel else None
                yscale = 'linear'
                ylim = ylim_norm
            elif kind == 'normalizedInce':
                df = dataframes['df_plateau']
                ylabel = 'Spectrum [%]' if ylabel else None
            elif kind  == 'absolute':
                df = dataframes['df_abs']
                ylabel = 'Spectrum 'r'[$\mu V^2/Hz$]' if ylabel else None
                ylim = ylim_abs
        elif kind.startswith('periodic'):
            df = dataframes['df_abs']
            x = 'fm_freqs'
            y_kind_dict = {'fm_psd_peak_fit': 'periodic',
                           'fm_psd_ap_fit': 'periodicAP',
                           'fm_fooofed_spectrum': 'periodicFULL'}
            ylim = ylim_per
            if kind == 'periodicBOTH':
                y_vals = ['fm_psd_peak_fit',
                          'fm_psd_ap_fit',
                          'fm_fooofed_spectrum']
                plot_lines = [False, True, True]  # plot ap + full
                plot_clusters = [False, True, True]  # very tricky. See line 52
                palette = [[None,
                            cfg.COLOR_DIC['periodicAP'],
                            cfg.COLOR_DIC['periodic']],
                           [None,
                            cfg.COLOR_DIC['periodicAP2'],
                            cfg.COLOR_DIC['periodic2']]]
                legend_palette = [[cfg.COLOR_DIC[y_kind_dict[y]]
                                   for y in y_vals],
                                  [cfg.COLOR_DIC[y_kind_dict[y] + '2']
                                   for y in y_vals]]
                # shift periodic power to make it visible
                df = df.copy()  # important to not alter df when shifting
                periodic_shift = 3
                df['fm_fooofed_spectrum'] += df['fm_fooofed_spectrum'] * \
                    periodic_shift
            else:
                palette = [[cfg.COLOR_DIC[kind]], [cfg.COLOR_DIC[kind + '2']]]
                legend_palette = palette
                if kind == 'periodic':
                    y_vals = ['fm_psd_peak_fit']
                    yscale = 'linear'
                elif kind == 'periodicAP':
                    y_vals = ['fm_psd_ap_fit']
                    ylim = ylim_ap
                elif kind == 'periodicFULL':
                    y_vals = ['fm_fooofed_spectrum']
            labels = [cfg.KIND_DICT[y_kind_dict[y]] for y in y_vals]
            ylabel = 'Spectrum 'r'[$\mu V^2/Hz$]' if ylabel else None
        else:
            raise ValueError(f"Unknown kind {kind}")
        if idx > 0:
            ylabel = None

        df_cond = df[(df.cond == cond) & (df.project == 'all')]
        labels_conds = [labels, labels]  # duplicate to avoid altering
        ic = 0 if cond == 'off' else 1
        colors = palette[ic]
        legend_colors = legend_palette[ic]
        labels = labels_conds[ic]
        if cond == 'offon_abs':
            y_vals = [y + '_log' for y in y_vals]
        else:
            # for OFF and ON values, use linear units. Important: Plot and
            # statistics are different when performed on psd_log vs psd. The
            # stats are based on the mean of the psd. mean(psd_a) vs
            # mean(psd_b) on a log scale is different from mean(log(psd_a)) vs
            # mean(log(psd_b)). Linear values on log scale appear more
            # reasonable since any errors based on non-linearities can be
            # excluded.
            pass

            # drop nan arrays (df.dropna(y) does not work for arrays)
            for y in y_vals:
                check_nan = lambda x: np.all(np.isnan(x))
                mask = pd.Series(df_cond[y].apply(check_nan))
                df_cond = df_cond[~mask]

            if lateralized_updrs:
                updrs = 'UPDRS_bradyrigid_contra'
            else:
                updrs = 'UPDRS_III'
            if within_comparison:
                # Stat settings
                paired_x1x2 = True
                hue = f"patient_symptom_dominant_side_BR_{cond}"
                # remove subjects without asymmetry
                # rename rows based on dictionary
                rename = {'severe side': 'More affected',
                          'mild side': 'Less affected'}
                hue_order = ['More affected', 'Less affected']

                if cond == 'on':
                    # only include consistent asymmetry for ON subjects to
                    # exclude possible LDOPA side effects
                    df_cond = df_cond[df_cond.dominant_side_consistent]

            else:
                # Stat settings
                paired_x1x2 = False
                # rename to ease interpretation
                if cond in ['off', 'on']:
                    sampling = ('hemispheres' if lateralized_updrs
                                else 'patients')
                    rename = {'mild_half': f'Mild {sampling}',
                              'severe_half': f'Severe {sampling}'}
                    hue_order = [f'Severe {sampling}', f'Mild {sampling}']
                elif cond.startswith('offon'):
                    rename = {'mild_half': 'Weak responders',
                              'severe_half': 'Strong responders'}
                    hue_order = ['Weak responders', 'Strong responders']
                hue = f'{updrs}_severity_median'
            df_cond[hue] = df_cond[hue].astype('category')
            df_cond[hue] = df_cond[hue].cat.rename_categories(rename)
            df_cond[hue] = df_cond[hue].cat.remove_unused_categories()
            df_cond = df_cond[df_cond[hue].isin(hue_order)]

            if within_comparison:
                # Plot settings
                cluster_str = 'within'
                # does nothing if asymmetric subjects selected:
                df_cond = _remove_single_hemi_subs(df_cond)
            else:
                # Plot settings
                cluster_str = 'across'
                if not lateralized_updrs:
                    # average hemispheres
                    group = df_cond.groupby(['subject'])
                    df_cond[x] = group[x].transform("mean")
                    for y in y_vals:
                        df_cond[y] = group[y].transform("mean")
                    df_cond = df_cond.drop_duplicates(subset=["subject"])

            # Extract cluster varible
            cluster_conds = df_cond[hue].unique()
            assert len(cluster_conds) == 2

            df1 = df_cond[df_cond[hue] == cluster_conds[0]]
            df2 = df_cond[df_cond[hue] == cluster_conds[1]]

            if within_comparison:
                assert (df1.subject.to_numpy() == df2.subject.to_numpy()).all()

            # Plotting
            all_clusters = []
            periodic_stats = []  # here it gets complicated. I need to
            # transfer the periodic cluster stats to full plot in order have
            # both correct stats and a correct visualization.
            for i, y in enumerate(y_vals):
                x_array, y_array, times = _extract_arrays(df1, df2, x, y,
                                                          xmax, xmin)

                if plot_lines[i]:
                    # do not plot fitted peaks without aperiodic power
                    df_exploded = explode_df(df_cond, freqs=x, psd=y,
                                            keep_cols=[hue, 'sub_hemi'],
                                            fmax=xmax, fmin=xmin)

                    n_x = x_array.shape[1]
                    n_y = y_array.shape[1]
                    assert df_exploded[df_exploded[hue]
                                       == cluster_conds[0]
                                       ].sub_hemi.nunique() == n_x
                    assert df_exploded[df_exploded[hue]
                                       == cluster_conds[1]
                                       ].sub_hemi.nunique() == n_y

                    sns.lineplot(data=df_exploded, x=x, y=y, hue=hue, ax=ax,
                                 errorbar=CI_SPECT,
                                 hue_order=hue_order,
                                 palette=[colors[i], colors[i]],
                                 style=hue, style_order=hue_order)

                print(f'\n{kind} {cond} (<{xmax} Hz):\n', file=output_file)
                clusters, cluster_count, cluster_borders = _get_clusters(
                    x_array, y_array, times, output_file, n_perm=n_perm,
                    paired_x1x2=paired_x1x2
                    )

                all_clusters.append(cluster_borders)
                if y.startswith('fm_'):
                    periodic_stats.append((clusters, cluster_count))
                if plot_clusters[i]:
                    if y.startswith('fm_fooofed_spectrum'):
                        clusters, cluster_count = periodic_stats[0]
                    _plot_clusters(ax, x_array, y_array, times, clusters,
                                   cluster_count, color_cluster=color_cluster,
                                   alpha_plot=0.4)

            # Set axis
            ax.set_xscale(xscale)
            if xlabel:
                ax.set_xlabel("Frequency [Hz]")
            else:
                ax.set_xlabel(None)
            if xmax == 45:
                xticks = XTICKS_FREQ_low
                xticklabels = XTICKS_FREQ_low_labels
            elif xmax == 60:
                xticks = XTICKS_FREQ_high
                xticklabels = XTICKS_FREQ_high_labels_skip13
            elif xmax == 200:
                xticks = [xmin, 50, 200]
                xticklabels = xticks
            else:
                raise ValueError(f'Set xticks for {xmax} Hz.')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_xlim(xmin, xmax)
            if kind == 'absolute' and xmax == 200:
                ticks = [0.1, 1]
                ax.set_yticks(ticks=ticks)
                ax.xaxis.set_tick_params(which='major', pad=0)
                ax.yaxis.set_tick_params(which='major', pad=0)
            if within_comparison:
                assert n_x == n_y
                sample_size_str = f'{cfg.SAMPLE_STN}'r'$=2 \times$'f'{n_x}'
            else:
                if lateralized_updrs:
                    sample_size_str = f'{cfg.SAMPLE_STN}={n_x} vs {n_y}'
                else:
                    sample_size_str = f'{cfg.SAMPLE_PAT}={n_x} vs {n_y}'
            cond_str = cfg.COND_DICT[cond]
            if info_title:
                kind_str = cfg.KIND_DICT[kind]
                updrs_str = cfg.PLOT_LABELS[updrs]
                title = (f'{kind_str}, {cond_str}, {updrs_str}, '
                         f'{sample_size_str}')
                ax.set_title(title)
                info_str = '_LongTitle'
            elif info_title is None:
                ax.set_title(f'{cond_str}: {sample_size_str}')
                info_str = '_Title'
            elif info_title is False:
                info_str = ''
            ax.set_ylim(ylim)
            _annotate_stats(ax, all_clusters, legend_colors, labels,
                            yscale=yscale, height=stat_height)
            if legend:
                if kind == 'periodicBOTH':
                    handles = [Line2D([0], [0], color=color, ls='-')
                                for color in legend_colors]
                    # do not show cluster stats for full model
                    total_stats = False
                    if not total_stats:
                        labels = labels[:-1]
                        handles = handles[:-1]
                    title = None
                else:
                    # Plot neutral legend without kind colors
                    handles = [Line2D([0], [0], color='k', ls=ls)
                               for ls in ['-', '--']]
                    labels = hue_order
                    # Plot legend in kind colors
                    if within_comparison:
                        title = 'Hemisphere'
                    else:
                        title = None
                ax.legend(handles, labels, handlelength=1, title=title,
                          borderaxespad=0.1, **leg_kws)
                legend = False  # only on first axis
            else:
                ax.get_legend().remove()
        if kind in ['absolute', 'periodicAP']:
            yticks = [0.1, 1]
        elif kind == 'periodic':
            yticks = [0, 0.5, 1]
        elif kind == 'normalized':
            yticks = [0, 1, 2, 3, 4]
        else:
            raise ValueError(f"Set yticks for {kind}")
        if cond == 'on':
            yticklabels = ['' for _ in yticks]
        else:
            yticklabels = yticks
        ax.set_yscale(yscale)
        ax.set_yticks(yticks, labels=yticklabels)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
    save_dir = join(cfg.FIG_PAPER, fig_dir)
    cond_str = '_'.join(conds)
    kind_str = '_'.join(kinds)
    fname = (f'{prefix}psd_clusters_{cluster_str}_{kind_str}_'
             f'{hue}_{xmax}Hz_{cond_str}_'
            f'{xscale}_{yscale}'
            f'{info_str}')
    plt.tight_layout()
    plt.subplots_adjust(wspace=.2)
    _save_fig(fig, fname, save_dir, bbox_inches=None,
              facecolor=(1, 1, 1, 0))
