"""Helping functions."""
import math
from os.path import join

import numpy as np
import pingouin as pg

import scripts.config as cfg
from scripts.plot_figures.settings import *
from scripts.utils import _average_hemispheres
from scripts.utils_plot import _corr_results, _save_fig


def forest_plot_correlation(df, X, y, n_perm=N_PERM_CORR, r_lim=(-1, 1),
                            dataset_labels=True, add_sample_sizes=False,
                            significance_based_on_ci=True, prefix='',
                            band_titles=True, fig_dir='Figure2'):

    projects = cfg.PROJECT_NAMES
    n_projects = len(projects)
    cond = df.cond.unique()[0]

    # Plot
    n_plots = len(X)
    rows = 1
    cols = n_plots
    height = 1.2 if band_titles else 1.06
    width = 1 * n_plots
    project_order = list(reversed(projects))
    # Squeeze the first n-1 projects
    squeezed_y_positions = np.linspace(0.3, 1, n_projects - 1)
    last_y_position = 0  # Place the last project further down
    positions = np.concatenate(([last_y_position], squeezed_y_positions))

    fig, axes = plt.subplots(rows, cols, figsize=(width, height), sharey=True,
                             sharex=True)
    axes = np.array([axes]) if n_plots == 1 else axes

    labels = []
    colors = []
    for j, x in enumerate(X):
        ax = axes[j]
        for i, project in enumerate(project_order):
            # Calc correlation
            df_proj = df[df.project_nme == project]
            if 'contra' not in y:
                df_proj = _average_hemispheres(df_proj, x, y)
            corr_results = _corr_results(df_proj, x, y, 'spearman',
                                         n_perm=n_perm, pval_string=False)
            rho, sample_size, pval, _, ci = corr_results
            if j == 0:
                if add_sample_sizes:
                    label = project + f' (n={sample_size})'
                else:
                    label = project
                    if project == 'all':
                        sample_size_all = sample_size
                        print(f'Sample size for {y} {cond} (all): '
                              f'{sample_size_all}')
                labels.append(label)

            mean_val = rho
            ci_lower, ci_upper = ci
            error = [[mean_val - ci_lower], [ci_upper - mean_val]]
            markersize = sample_size / 20
            color = cfg.COLOR_DIC[project]
            colors.append(color)

            # Issue: CI ovlerapping with zero (insignificant) and
            # p-value < 0.05 (significant) is no contradiction since
            # statistical tests not equivalent. Can lead to minor
            # discrepancies. Solution: Use CI to indicate significance as
            # triangle/square marker since more intuitive and very similar
            # to p-value anyways.
            if significance_based_on_ci:
                # Test significance using bootstrap CI
                corr_positive = ci[0] > 0 and ci[1] > 0
                corr_negative = ci[0] < 0 and ci[1] < 0
                non_significant = ci[0] <= 0 <= ci[1]
                if corr_positive or corr_negative:
                    if corr_positive:
                        marker = '>'
                    elif corr_negative:
                        marker = '<'
                else:
                    marker = 's'
                    assert non_significant or np.isnan(rho)
            else:
                # Test significance using permutation test pvalue
                if pval < 0.05:
                    if rho > 0:
                        marker = '>'
                    elif rho < 0:
                        marker = '<'
                else:
                    marker = 's'

            if project == 'all':
                rho_all = rho
                ci_all = ci
                pval_all = pval

            # Plot markers
            error_kwargs = dict(xerr=error)
            x_plot = mean_val
            y_plot = positions[i]
            ax.plot(x_plot, y_plot, marker, color=color,
                    markersize=markersize)
            # Plot confidence intervals (bootstrap)
            ax.errorbar(x_plot, y_plot, **error_kwargs, color='grey',
                        markersize=0, capsize=0)

        # Power analysis
        required_n = pg.power_corr(r=rho_all, power=0.8)
        if ci_all[0] < 0 < ci_all[1]:
            required_n = ''
            # labelpad important to enable equal height with UPDRS-III if no
            # required n
            labelpad = 12.5
        else:
            if 'contra' not in y:
                required_n = (f'\nRequired {cfg.SAMPLE_PAT}'
                              f'={math.ceil(required_n)}')
            else:
                required_n = (f'\nRequired  {cfg.SAMPLE_STN}'
                              f'={math.ceil(required_n)}')
            labelpad = None

        # Set axis
        if band_titles:
            band_title = cfg.PLOT_LABELS[x].replace(' Peak',
                                                    '').replace(' mean', '')
        else:
            band_title = ''
        rho_string = r'$\rho_{\mathrm{all}}=$'f'{rho_all:.2f}'
        pval_string = (f"(p={pval_all:1.0e})" if abs(pval_all) < 0.005
                       else f"(p={pval_all:.2f})")
        fontweight = 'bold' if pval_all < 0.05 else 'normal'
        stat_string = rho_string + ' ' + pval_string + required_n
        ax.axvline(0, color="k", linestyle="-", lw=LINEWIDTH_AXES)
        if j > 0:
            ax.yaxis.set_tick_params(width=0, length=0)
        ax.spines['left'].set_visible(False)
        if dataset_labels:
            ticks = positions
            ax.set_yticks(ticks=ticks, labels=labels)
            for lab, color in zip(ax.get_yticklabels(), colors):
                lab.set_color(color)
        else:
            ax.set_yticks([])
        ax.set_xlabel(stat_string, fontweight=fontweight, ha='left', x=0,
                      labelpad=labelpad)
        title = band_title
        ax.set_title(title, linespacing=1.3, y=.9)
        ax.set_xlim(r_lim)
        ymin, ymax = ax.get_ylim()
        yscale = ymax - ymin
        ax.set_ylim(-yscale*.2, ymax*1.01)
    if df.psd_kind.unique()[0] == 'normalized':
        kind = 'normalized'
    elif 'fm' in x:
        kind = 'periodic'
    else:
        kind = 'absolute'
    figname = (f'{prefix}dataset_correlation_{cond}_{kind}_{y}_'
               f'nperm={n_perm}_samplesize={sample_size_all}')
    plt.tight_layout()
    _save_fig(fig, join(fig_dir, figname),
              cfg.FIG_PAPER, transparent=True, bbox_inches=None)
