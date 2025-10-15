"""Helping functions."""
import warnings
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import seaborn as sns
from pte_stats import cluster, timeseries
from scipy.stats import bootstrap

import scripts.config as cfg
from scripts.plot_figures.settings import (BANDS, CI_SPECT, LINEWIDTH_AXES,
                                           N_BOOT_COHEN, N_PERM_CLUSTER,
                                           XTICKS_FREQ_low,
                                           XTICKS_FREQ_low_labels)
from scripts.utils_plot import (_add_band, _axes2d, _save_fig, cohen_d,
                                equalize_x_and_y, explode_df)


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


def _get_clusters(x1, x2, times, alpha_sig=0.05, n_perm=100000,
                  paired_x1x2=True):
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
    cluster_vals = []
    for cluster_idx in range(1, cluster_count + 1):
        index = np.where(clusters == cluster_idx)[0]
        if index.size == 0:
            continue
        lims = np.arange(index[0], index[-1] + 1)
        time_0 = x_labels[lims[0]]
        time_1 = x_labels[lims[-1]]
        cluster_times.append((time_0, time_1))
        cluster_vals.append(data_a_stat[lims].mean())
    cluster_borders = np.array(cluster_times)
    return clusters, cluster_count, cluster_borders, cluster_vals


def _plot_clusters_line(ax, times, clusters, cluster_count, height, color):
    for cluster_idx in range(1, cluster_count + 1):
        index = np.where(clusters == cluster_idx)[0]
        if index.size == 0:
            continue
        lims = np.arange(index[0], index[-1] + 1)
        y_arr = np.ones((times[lims].shape[0], 1)) * height
        ax.plot(times[lims], y_arr, color=color, lw=1)


def _mean_psds_ax(ax, df_proj, x, y, kind, hue,
                  add_bands=False, scale='log', fm_params=False,
                  n_perm=10000, paired=True, ylabel=True,
                  palette=None, xmin=2, xmax=45, output_file=None):
    proj = df_proj.project.unique()[0]
    proj_nme = df_proj.project_nme.unique()[0]

    if hue == 'cond':
        hue_order = ['off', 'on']
        df_proj = df_proj[df_proj[hue].isin(hue_order)]
        if palette is None:
            palette = [cfg.COLOR_DIC[proj], cfg.COLOR_DIC[proj + "2"]]
        else:
            palette = [cfg.COLOR_DIC[kind], cfg.COLOR_DIC[kind + '2']]
        style = style_order = None
        keep_cols = ['fm_params']
        if kind == 'normalized':
            height = 0.1
        else:
            height = -.01
    elif hue.endswith('_severity_median'):
        assert df_proj.cond.nunique() == 1, "Don't mix conds"
        if hue == 'UPDRS_III_severity_median':
            # average hemispheres
            group = df_proj.groupby(['subject'])
            df_proj.loc[:, x] = group[x].transform("mean")
            df_proj.loc[:, y] = group[y].transform("mean")
            df_proj = df_proj.drop_duplicates(subset=["subject"])
            sampling = 'patients'
        elif hue == 'UPDRS_bradyrigid_contra_severity_median':
            sampling = 'STNs'
        n_mild = df_proj[df_proj[hue] == 'mild_half'].shape[0]
        n_severe = df_proj[df_proj[hue] == 'severe_half'].shape[0]
        sample_size_mild = f'(n={n_mild})'
        sample_size_severe = f'(n={n_severe})'
        mild_nme = f'Mild {sampling} {sample_size_mild}'
        severe_nme = f'Severe {sampling} {sample_size_severe}'
        rename = {'mild_half': mild_nme, 'severe_half': severe_nme}
        hue_order = [severe_nme, mild_nme]
        style = hue
        style_order = hue_order
        keep_cols = ['fm_params', hue]
        df_proj[hue].astype(str).replace(rename, inplace=True)
        df_proj.loc[:, hue] = df_proj[hue].astype('category')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            df_proj.loc[:, hue] = df_proj[hue].cat.rename_categories(rename)
        df_proj.loc[:, hue] = df_proj[hue].cat.remove_unused_categories()
        df_proj = df_proj[df_proj[hue].isin(hue_order)]
        height = -.01

    df_psd = explode_df(df_proj, freqs=x, psd=y, fm_params=fm_params,
                        keep_cols=keep_cols, fmin=xmin, fmax=xmax)

    sns.lineplot(data=df_psd, ax=ax, x=x, y=y, hue=hue, palette=palette,
                 hue_order=hue_order, style=style, style_order=style_order,
                 errorbar=CI_SPECT)
    handles, labels = ax.get_legend_handles_labels()
    if labels[0] in cfg.PLOT_LABELS:
        labels = [cfg.PLOT_LABELS[label] for label in labels]

    # Cluster statistics

    # Extract cluster variable
    cluster_conds = df_proj[hue].unique()
    msg = f'Only two conds supported got {cluster_conds}'
    assert len(cluster_conds) == 2, msg

    df1 = df_proj[df_proj[hue] == cluster_conds[0]]
    df2 = df_proj[df_proj[hue] == cluster_conds[1]]

    if paired:
        assert (df1.subject.to_numpy() == df2.subject.to_numpy()).all()
        assert (df1.subject.values == df2.subject.values).all()

    x_array, y_array, times = _extract_arrays(df1, df2, x, y, xmax, xmin)

    clusters, cluster_count, cluster_borders, cluster_vals = _get_clusters(
            x_array, y_array, times, n_perm=n_perm, paired_x1x2=paired)

    ldopa_enhanced_clusters = []
    ldopa_reduced_clusters = []
    for idx, border in enumerate(cluster_borders):
        if cluster_vals[idx] > 0:
            ldopa_reduced_clusters.append(border)
        elif cluster_vals[idx] < 0:
            ldopa_enhanced_clusters.append(border)
        else:
            raise ValueError("Cluster value should not be zero")
    print('\n', file=output_file)
    print(f'{proj_nme} clusters:\n', file=output_file)
    print('Enhanced:', file=output_file)
    for cluster in ldopa_enhanced_clusters:
        print(f'{cluster[0]:.0f}-{cluster[1]:.0f} Hz', file=output_file)
    print('\n', file=output_file)
    print('Reduced:', file=output_file)
    for cluster in ldopa_reduced_clusters:
        print(f'{cluster[0]:.0f}-{cluster[1]:.0f} Hz', file=output_file)
    print('\n', file=output_file)

    _plot_clusters_line(ax, times, clusters, cluster_count, height, palette[0])

    if ylabel:
        if proj == 'Neumann' or kind in ['absolute', 'periodic', 'lorentzian']:
            if scale == 'linear':
                if kind in ['normalized', 'relative']:
                    ylabel = "Normalized spectra "r"[$\%$]"
                else:
                    ylabel = 'Spectra 'r'[$\mu V^2/Hz$]'
            else:
                if kind in ['normalized', 'relative']:
                    ylabel = "Normalized spectra\n[log10(%)]"
                else:
                    ylabel = 'Spectra 'r'[$log10(\mu V^2/Hz)$]'
    else:
        ylabel = None
    _add_band(add_bands, ax)
    if proj == 'all' or kind in ['normalized', 'relative']:
        ax.legend(handles, labels, loc='upper right', title=None,
                  handlelength=1, fontsize=6)
    else:
        ax.get_legend().remove()
    ax.set_ylabel(ylabel)
    if scale == 'linear':
        if kind in ['normalized', 'relative']:
            ylim = (0, 10)
        else:
            ylim = (-.02, 1.2)
            ax.set_yticks([0, .2, .4, .6, .8, 1, 1.2])
    ax.set_ylim(ylim)
    ax.set_xticks(XTICKS_FREQ_low)
    ax.set_xticklabels(XTICKS_FREQ_low_labels)
    buffer = .2
    ax.set_xlim(XTICKS_FREQ_low[0] - buffer, XTICKS_FREQ_low[-1] + buffer)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)
    return df_proj, hue_order


def _forrest_plot_ax(ax, df_proj, kind, hue, bands=BANDS,
                     estimator='effect_size',
                     add_band_colors=False, scale='log', hue_order=None,
                     paired=True, ylabel=True, height_star=1,
                     n_boot=10000, palette=None, output_file=None):
    proj = df_proj.project.unique()[0]
    proj_nme = df_proj.project_nme.unique()[0]

    df_proj = df_proj[df_proj[hue].isin(hue_order)]
    assert df_proj[hue].nunique() == 2, 'Effect size requires two conds'

    # get power columns and band means
    if kind in ['normalized', 'absolute']:
        pwr = '_abs_mean_log'
    elif kind in ['periodic', 'lorentzian']:
        pwr = '_fm_mean_log'

    if scale == 'linear':
        pwr = pwr.replace('_log', '')

    band_borders = []
    band_means = []
    values = []
    errors = []
    significances = []
    print(f'{proj_nme} {estimator}:\n', file=output_file)
    for band in bands:
        pwr_col = band + pwr
        if band == 'gamma_broad':
            # don't plot gamma outside of plotting range
            band_border = (cfg.BANDS[band][0], 45)
        else:
            band_border = cfg.BANDS[band]
        band_borders.extend(band_border)
        band_means.append(np.mean(band_border))
        arr1 = df_proj[df_proj[hue] == hue_order[0]][pwr_col].values
        arr2 = df_proj[df_proj[hue] == hue_order[1]][pwr_col].values
        nx = len(arr1)
        ny = len(arr2)
        if estimator == 'effect_size':
            value = cohen_d(arr1, arr2)
            if np.all(arr1 == 0) or np.all(arr2 == 0):
                # effect size unreasonable if all values are zero
                value = np.nan
            data = (arr1, arr2)
            func = cohen_d
        else:
            if estimator == 'mean':
                func = np.mean
            elif estimator == 'median':
                func = np.median
            else:
                raise ValueError('Unknown estimator')
            value = func(arr1 - arr2)
            data = (arr1 - arr2,)
        print(f'{band}: {value:.2f}', file=output_file)
        values.append(value)
        # Multiple comparisons corrected confidence intervals
        alpha = 0.05
        multiple_comparisons = len(bands)
        confidence_level = 1 - (alpha / multiple_comparisons)

        if n_boot is None:
            msg = 'Only effect size supports parametric'
            assert estimator == 'effect_size', msg
            # parametric (fast but more assumptions)
            ci = pg.compute_esci(value, nx, ny, paired=paired,
                                 confidence=confidence_level)
        else:
            # nonparametric (slow but more correct)
            result = bootstrap(data, func, n_resamples=n_boot, paired=paired,
                               confidence_level=confidence_level,
                               random_state=1)
            ci = result.confidence_interval
            # if distribution is degenerate (many 0 values), apply parametric
            if np.any(np.isnan(result.bootstrap_distribution)):
                ci = pg.compute_esci(value, nx, ny, paired=paired)

        # CIs
        ci_lower, ci_upper = ci
        error = [[abs(value - ci_lower)], [abs(ci_upper - value)]]
        errors.append(error)
        # make sure that np.nan values are not significant
        if value is np.nan:
            significant = False
        elif ci_lower < 0 and ci_upper < 0 or ci_lower > 0 and ci_upper > 0:
            significant = True
        else:
            significant = False
        significances.append(significant)

    # Plot CIs
    yerr = np.array(errors).reshape(len(bands), 2).T
    if palette is None:
        color = cfg.COLOR_DIC[proj]
    else:
        color = palette[0]
    ax.errorbar(band_means, values, yerr=yerr,
                color=color,  # looks much better than 'k'
                ls='',  # do not plot connecting lines
                markersize=0, capsize=0)

    # Plot effect sizes as bars
    values = np.repeat(values, 2)  # duplicate values for bars
    ax.plot(band_borders, values, c='k', lw=LINEWIDTH_AXES)
    for band_border, effect_size in zip(band_borders, values):
        ax.vlines(band_border, 0, effect_size, color='k', lw=LINEWIDTH_AXES)
    for j in range(0, len(values)-1, 2):
        border1 = band_borders[j]
        border2 = band_borders[j + 1]
        effect1 = values[j]
        effect2 = values[j + 1]
        borders = [border1, border2]
        effects = [effect1, effect2]
        band = BANDS[j//2]
        if add_band_colors:
            color = cfg.BAND_COLORS[band] if proj == 'all' else 'grey'
        else:
            color = 'grey'
        ax.fill_between(borders, effects, color=color, alpha=.4)
    for j, sig in enumerate(significances):
        if sig:
            display_coords = ax.transData.transform((band_means[j], 0))
            axes_coords = ax.transAxes.inverted().transform(display_coords)
            x_pos = axes_coords[0]
            # convert pval to asterisk
            ax.text(x_pos, height_star, '*', ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=9, color='k')

    ax.axhline(0, color="k", lw=LINEWIDTH_AXES)

    # set axis
    if hue == 'cond':
        if proj != 'all':
            ax.set_xticks(XTICKS_FREQ_low[:-1],
                          labels=XTICKS_FREQ_low_labels[:-1])
            xmax = XTICKS_FREQ_low[-1]
        elif proj == 'all':
            ax.set_xticks(XTICKS_FREQ_low, labels=XTICKS_FREQ_low_labels)
            xmax = XTICKS_FREQ_low[-1]
    else:
        ax.set_xticks(XTICKS_FREQ_low, labels=XTICKS_FREQ_low_labels)
        xmax = XTICKS_FREQ_low[-1]
    xmin = XTICKS_FREQ_low[0]
    # add tiny buffer to exactly match white grid line to plot border
    buffer = 0.09
    xmin -= buffer
    xmax += buffer
    ax.set_xlim(xmin, xmax)
    if kind in ['absolute', 'periodic']:
        ax.set_yticks([-.3, 0, .3])
        ax.set_ylim([-.6, .6])
    elif kind in ['normalized']:
        ax.set_yticks([-1, -.5, 0, .5, 1], labels=[-1, '', 0, '', 1])
        ax.set_ylim([-1, 1])
    if ylabel:
        if hue == 'cond':
            ylabel = (f'Power {cfg.COND_DICT['off']}-{cfg.COND_DICT['on']} '
                      r'[$d$]')
        else:
            ylabel = 'Severe - mild'
    else:
        ylabel = None
    ax.set_ylabel(ylabel)


def plot_normalized_spectra(df_norm, fig_dir='Figure1', prefix='',
                            n_perm_cluster=N_PERM_CLUSTER,
                            n_boot_cohen=N_BOOT_COHEN):
    estimator = 'effect_size'
    kind = 'normalized'
    paired = True
    psd = 'psd'
    freqs = 'psd_freqs'
    fm_params = False
    projects = cfg.PROJECT_ORDER_SLIM
    palette = None
    scale = 'linear'

    hue = 'cond'
    n_cols = len(projects)
    n_rows = 2
    fig_x_size = 7.16 if n_cols > 1 else 1.5

    # Create output directory
    output_dir = join(cfg.FIG_PAPER, fig_dir)

    # Open an output file for saving text logs
    output_file_path = join(output_dir, f"{prefix}_output.txt")

    with open(output_file_path, "w") as output_file:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_x_size, 2),
                                 sharex="col", sharey="row",
                                 height_ratios=[3, 1.5])
        axes = _axes2d(axes, n_rows, n_cols)
        for i, proj in enumerate(projects):
            df_proj = df_norm[df_norm.project == proj]
            if paired:
                # smaller sample size but stronger statistics
                df_proj, n_sample = equalize_x_and_y(df_proj, 'cond', psd)
            else:
                n_sample = df_proj.subject.nunique()
            bands = False
            ax = axes[0, i]
            df_proj, hue_order = _mean_psds_ax(ax, df_proj, freqs, psd, kind,
                                               hue, paired=paired,
                                               fm_params=fm_params,
                                               n_perm=n_perm_cluster,
                                               add_bands=bands, scale=scale,
                                               palette=palette,
                                               output_file=output_file)
            ax = axes[1, i]
            ylabel = True if i == 0 else False
            _forrest_plot_ax(ax, df_proj, kind, hue, estimator=estimator,
                             hue_order=hue_order, paired=paired,
                             n_boot=n_boot_cohen,
                             ylabel=ylabel, add_band_colors=bands, scale=scale,
                             palette=palette, output_file=output_file)
            sample_size_str = f' ({cfg.SAMPLE_STN}='f"{n_sample})"
            axes[0, i].set_title(cfg.PROJECT_DICT[proj] + sample_size_str,
                                 color=cfg.COLOR_DIC[proj])
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1, wspace=0.05)
        fig_name = f'{prefix}PSDs_{estimator}s_{kind}_{scale}_paired={paired}'
        _save_fig(fig, join(fig_dir, fig_name), cfg.FIG_PAPER,
                  transparent=False, bbox_inches=None)


def plot_absolute_spectra(df_abs, fig_dir='Figure_S1', prefix='',
                          n_perm=N_PERM_CLUSTER, n_boot=N_BOOT_COHEN):
    estimator = 'effect_size'
    kind = 'absolute'
    paired = True
    psd = 'psd'
    freqs = 'psd_freqs'
    fm_params = False
    projects = cfg.PROJECT_ORDER_SLIM
    palette = None
    scale = 'linear'

    hue = 'cond'
    n_cols = len(projects)
    n_rows = 2
    fig_x_size = 7.2 if n_cols > 1 else 1.5

    # Create output directory
    output_dir = join(cfg.FIG_PAPER, fig_dir)

    # Open an output file for saving text logs
    output_file_path = join(output_dir, f"{prefix}_output.txt")

    with open(output_file_path, "w") as output_file:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_x_size, 2),
                                 sharex="col", sharey="row",
                                 height_ratios=[3, 1.5])
        axes = _axes2d(axes, n_rows, n_cols)
        for i, proj in enumerate(projects):
            df_proj = df_abs[df_abs.project == proj]
            if paired:
                # smaller sample size but stronger statistics
                df_proj, n_sample = equalize_x_and_y(df_proj, 'cond', psd)
            else:
                n_sample = df_proj.subject.nunique()
            bands = False
            ax = axes[0, i]
            df_proj, hue_order = _mean_psds_ax(ax, df_proj, freqs, psd, kind,
                                               hue, paired=paired,
                                               fm_params=fm_params,
                                               n_perm=n_perm,
                                               add_bands=bands, scale=scale,
                                               palette=palette,
                                               output_file=output_file)
            ax = axes[1, i]
            ylabel = True if i == 0 else False
            _forrest_plot_ax(ax, df_proj, kind, hue, estimator=estimator,
                             hue_order=hue_order, paired=paired,
                             n_boot=n_boot, ylabel=ylabel,
                             add_band_colors=bands, scale=scale,
                             palette=palette, output_file=output_file)
            sample_size_str = f' ({cfg.SAMPLE_STN}='f"{n_sample})"
            axes[0, i].set_title(cfg.PROJECT_DICT[proj] + sample_size_str,
                                 color=cfg.COLOR_DIC[proj])
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1, wspace=0.05)
        fig_name = f'{prefix}PSDs_{estimator}s_{kind}_{scale}_paired={paired}'
        _save_fig(fig, join(fig_dir, fig_name), cfg.FIG_PAPER,
                  transparent=False, bbox_inches=None)


def plot_abs_per_spectra(dataframes, kind, fig_dir='Figure4', prefix='',
                         height_star=1,
                         n_perm=N_PERM_CLUSTER, n_boot=N_BOOT_COHEN):
    estimator = 'effect_size'
    hues = ['cond', 'UPDRS_III_severity_median']
    proj = 'all'
    n_cols = len(hues)
    n_rows = 2
    fig_x_size = 3.35
    bands = False
    scale = 'linear'

    # Create output directory
    output_dir = join(cfg.FIG_PAPER, fig_dir)
    output_file_path = join(output_dir, f"{prefix}_output.txt")

    with open(output_file_path, "w") as output_file:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_x_size, 1.75),
                                 sharex="col", sharey="row",
                                 height_ratios=[3, 1.5])
        axes = _axes2d(axes, n_rows, n_cols)
        for col, hue in enumerate(hues):
            if hue == 'cond':
                paired = True
                conds = ['off', 'on']
            elif hue.endswith('_severity_median'):
                conds = ['off']
                paired = False
            if conds == ['off']:
                palette = [cfg.COLOR_DIC[kind], cfg.COLOR_DIC[kind]]
            elif conds == ['on']:
                palette = [cfg.COLOR_DIC[kind + '2'],
                           cfg.COLOR_DIC[kind + '2']]
            elif conds == ['off', 'on']:
                palette = [cfg.COLOR_DIC[kind], cfg.COLOR_DIC[kind + '2']]
            else:
                palette = None
            if kind == 'absolute':
                df_proj = dataframes['df_abs']
                psd = 'psd'
                df_proj = df_proj[(df_proj.project == proj)
                                  & (df_proj.cond.isin(conds))]
                if paired:
                    # smaller sample size but stronger statistics
                    df_proj, n_sample = equalize_x_and_y(df_proj, hue, psd)
                else:
                    n_sample = df_proj.subject.nunique()
                freqs = 'psd_freqs'
            elif kind == 'periodic':
                df_proj = dataframes['df_per']
                df_proj = df_proj[(df_proj.project == proj)
                                  & (df_proj.cond.isin(conds))]
                psd = 'fm_psd_peak_fit'
                freqs = 'fm_freqs'
                if paired:
                    # smaller sample size but stronger statistics
                    df_proj, n_sample = equalize_x_and_y(df_proj, hue, psd)
                else:
                    n_sample = df_proj.subject.nunique()
            else:
                raise ValueError(f'Unknown kind {kind}')

            if scale == 'log':
                psd += '_log'

            ax = axes[0, col]
            if hue == 'cond':
                sample_size_stn = n_sample
            df_proj, hue_order = _mean_psds_ax(ax, df_proj, freqs, psd, kind,
                                               hue,
                                               paired=paired,
                                               n_perm=n_perm,
                                               add_bands=bands,
                                               ylabel=False,
                                               scale=scale, palette=palette,
                                               output_file=output_file)
            ax = axes[1, col]
            _forrest_plot_ax(ax, df_proj, kind, hue, estimator=estimator,
                             paired=paired,
                             hue_order=hue_order,
                             height_star=height_star,
                             n_boot=n_boot, add_band_colors=bands,
                             scale=scale, ylabel=False,
                             palette=palette, output_file=output_file)
            ax.set_xlabel(None)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1, wspace=None)
        fig_name = (f'{prefix}PSDs_{estimator}_{kind}_{scale}_'
                    f'{'_'.join(hues)}_samplesize={sample_size_stn}')
        _save_fig(fig, join(fig_dir, fig_name), cfg.FIG_PAPER,
                  transparent=False, bbox_inches=None)
